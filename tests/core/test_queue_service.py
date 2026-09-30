"""QueueService tests with a fake launcher: fake phase handles that answer like a child."""

import asyncio
import subprocess
import sys

import pytest
from aTrain_core import outputs
from aTrain_core.jobs import JobStatus, JobStore, QueueLockedError, Step
from aTrain_core.queue_service import QueueService
from aTrain_core.runner import (
    RAW_CHECKPOINT,
    JobDone,
    JobFailed,
    JobFileId,
    JobProgress,
    JobTranscribed,
    ModelLoaded,
    NextJob,
    NoMoreWork,
    PhaseDied,
    PhaseFinished,
)
from aTrain_core.settings import ComputeType, Device
from tests.unit.test_jobs import make_spec


class FakeHandle:
    """Behaves like a phase child. Per job id, `outcomes` says what happens:
    "done" (default), "fail", "crash", "done_then_die" (JobDone is in the pipe when it is
    killed), or "hang" (works until killed). A job with a gate waits for it first."""

    def __init__(self, launcher, phase, arg):
        self.launcher, self.phase, self.arg = launcher, phase, arg
        self.replies: asyncio.Queue = asyncio.Queue()
        self.dispatched: list[str] = []

    async def events(self):
        launcher = self.launcher
        if self.phase == 1 and self.arg.model in launcher.load_crash_models:
            yield PhaseDied(1)
            return
        yield ModelLoaded()
        while True:
            yield NextJob()
            reply = await self.replies.get()
            if reply is None:
                yield PhaseDied(-9)
                return
            if isinstance(reply, NoMoreWork):
                yield PhaseFinished()
                return
            job = reply.job
            job_id = job.spec.id
            self.dispatched.append(job_id)
            if job_id in launcher.gates:
                await launcher.gates[job_id].wait()
            outcome = launcher.outcomes.get(job_id, "done")
            yield JobProgress(job_id, "Transcribe", 1, 2)
            if outcome == "fail":
                yield JobFailed(job_id, Step.TRANSCRIPTION, "model error", "Traceback")
                continue
            if outcome == "crash":
                yield PhaseDied(1)
                return
            if outcome == "hang":
                await self.replies.get()  # until killed
                yield PhaseDied(-9)
                return
            if self.phase == 1 and job.spec.speaker_detection:
                job.work_dir.mkdir(parents=True, exist_ok=True)
                outputs.write_checkpoint(
                    job.work_dir / RAW_CHECKPOINT,
                    transcript={"segments": []},
                    audio_duration=5,
                    source=job.spec.source,
                )
                yield JobTranscribed(job_id, 5)
                continue
            file_id = f"{job_id}-archive"
            (outputs.TRANSCRIPT_DIR / file_id).mkdir(parents=True)
            yield JobFileId(job_id, file_id)
            yield JobDone(job_id, 5, [])
            if outcome == "done_then_die":
                await self.replies.get()
                yield PhaseDied(-9)
                return

    def reply(self, answer):
        self.replies.put_nowait(answer)

    def kill(self):
        self.replies.put_nowait(None)


class FakeLauncher:
    def __init__(self):
        self.launches: list[tuple[int, object]] = []
        self.handles: list[FakeHandle] = []
        self.outcomes: dict[str, str] = {}
        self.gates: dict[str, asyncio.Event] = {}
        self.load_crash_models: set[str] = set()

    def launch(self, phase):
        def launch(arg):
            self.launches.append((phase, arg))
            handle = FakeHandle(self, phase, arg)
            self.handles.append(handle)
            return handle

        return launch

    def dispatched(self) -> list[str]:
        return [job_id for handle in self.handles for job_id in handle.dispatched]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs, "TRANSCRIPT_DIR", tmp_path / "transcriptions")
    launcher = FakeLauncher()
    store = JobStore(tmp_path / "queue")
    service = QueueService(
        store, launch_phase1=launcher.launch(1), launch_phase2=launcher.launch(2)
    )
    return tmp_path, store, service, launcher


def spec(tmp_path, job_id, **overrides):
    source = tmp_path / f"{job_id}.mp3"
    source.write_bytes(b"audio " + job_id.encode())
    values = {"device": Device.CPU, "compute_type": ComputeType.INT8, "speaker_detection": False}
    values.update(overrides)
    return make_spec(id=job_id, source=source, display_name=source.name, export_dir=None, **values)


def status(store, job_id):
    return store.get(job_id)[1].status


async def until(condition, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


async def settle(service):
    await until(
        lambda: all(
            state.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}
            for _, state in service.jobs()
        )
    )


@pytest.fixture
async def started(env):
    _tmp_path, _store, service, _launcher = env
    await service.start()
    yield env
    await service.stop()


async def test_job_without_speaker_detection_is_done_in_phase_1(started):
    tmp_path, store, service, launcher = started
    staged = store.uploads_root / "a" / "a.mp3"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"audio")
    store.work_dir("a").mkdir(parents=True)
    job = spec(tmp_path, "a")
    service.enqueue([job.__class__.from_json({**job.to_json(), "source": str(staged)})])
    await settle(service)

    state = store.get("a")[1]
    assert state.status == JobStatus.DONE and state.file_id == "a-archive"
    assert [phase for phase, _ in launcher.launches] == [1]
    assert not store.work_dir("a").exists() and not staged.parent.exists()


async def test_speaker_detection_runs_phase_2_after_the_group(started):
    tmp_path, store, service, launcher = started
    service.enqueue(
        [spec(tmp_path, "a", speaker_detection=True), spec(tmp_path, "b", speaker_detection=True)]
    )
    await settle(service)

    assert [phase for phase, _ in launcher.launches] == [1, 2]
    assert launcher.dispatched() == ["a", "b", "a", "b"]
    assert status(store, "a") == status(store, "b") == JobStatus.DONE


async def test_grouping_follows_the_head_of_the_queue(started):
    tmp_path, _store, service, launcher = started
    service.pause()
    service.enqueue([spec(tmp_path, "a"), spec(tmp_path, "b", model="small"), spec(tmp_path, "c")])
    service.resume()
    await settle(service)

    assert [arg.model for _, arg in launcher.launches] == ["large-v3-turbo", "small"]
    assert launcher.dispatched() == ["a", "c", "b"]


async def test_cancelled_job_of_a_running_group_is_never_handed_out(started):
    tmp_path, store, service, launcher = started
    launcher.gates["a"] = asyncio.Event()
    service.enqueue([spec(tmp_path, "a"), spec(tmp_path, "b"), spec(tmp_path, "c")])
    await until(lambda: launcher.dispatched() == ["a"])

    await service.cancel(["b"])
    service.move("c", -1)
    launcher.gates["a"].set()
    await settle(service)

    assert launcher.dispatched() == ["a", "c"]
    assert status(store, "b") == JobStatus.CANCELLED


async def test_job_added_during_the_phase_joins_the_group(started):
    tmp_path, _store, service, launcher = started
    launcher.gates["a"] = asyncio.Event()
    service.enqueue([spec(tmp_path, "a")])
    await until(lambda: launcher.dispatched() == ["a"])
    service.enqueue([spec(tmp_path, "b")])
    launcher.gates["a"].set()
    await settle(service)

    assert len(launcher.launches) == 1 and launcher.dispatched() == ["a", "b"]


async def test_cancel_the_running_job(started):
    tmp_path, store, service, launcher = started
    launcher.outcomes["a"] = "hang"
    service.enqueue([spec(tmp_path, "a"), spec(tmp_path, "b")])
    await until(lambda: status(store, "a") == JobStatus.TRANSCRIBING)

    await service.cancel(["a"])
    await settle(service)

    assert status(store, "a") == JobStatus.CANCELLED and not store.get("a")[1].cancelling
    assert status(store, "b") == JobStatus.DONE
    assert len(launcher.launches) == 2  # b ran in a new child


async def test_job_that_finished_before_the_kill_stays_done(started):
    tmp_path, store, service, launcher = started
    launcher.outcomes["a"] = "done_then_die"
    launcher.gates["a"] = asyncio.Event()
    service.enqueue([spec(tmp_path, "a")])
    await until(lambda: launcher.dispatched() == ["a"])

    await service.cancel(["a"])  # the kill is requested while the job is in flight...
    assert store.get("a")[1].cancelling
    launcher.gates["a"].set()  # ...but JobDone was already on its way
    await until(
        lambda: launcher.handles[0].replies.empty() and status(store, "a") == JobStatus.DONE
    )

    assert status(store, "a") == JobStatus.DONE and not store.get("a")[1].cancelling
    assert (outputs.TRANSCRIPT_DIR / "a-archive").is_dir()


async def test_interrupted_output_writing_leaves_no_archive_folder(started):
    tmp_path, store, service, launcher = started
    launcher.outcomes["a"] = "hang"
    service.enqueue([spec(tmp_path, "a")])
    await until(lambda: status(store, "a") == JobStatus.TRANSCRIBING)
    (outputs.TRANSCRIPT_DIR / "partial").mkdir(parents=True)
    store.update("a", file_id="partial")

    await service.cancel(["a"])
    await settle(service)

    assert not (outputs.TRANSCRIPT_DIR / "partial").exists() and store.get("a")[1].file_id is None


async def test_pause_and_resume(started):
    tmp_path, store, service, launcher = started
    service.pause()
    service.enqueue([spec(tmp_path, "a")])
    await asyncio.sleep(0.1)
    assert launcher.launches == []
    service.resume()
    await settle(service)
    assert status(store, "a") == JobStatus.DONE


async def test_missing_source_fails_and_the_next_job_runs(started):
    tmp_path, store, service, _launcher = started
    service.pause()
    service.enqueue([spec(tmp_path, "a"), spec(tmp_path, "b")])
    (tmp_path / "a.mp3").unlink()
    service.resume()
    await settle(service)

    assert store.get("a")[1].error == "Source file not found"
    assert status(store, "b") == JobStatus.DONE


async def test_failed_job_does_not_stop_the_group(started):
    tmp_path, store, service, launcher = started
    launcher.outcomes["a"] = "fail"
    service.enqueue([spec(tmp_path, "a"), spec(tmp_path, "b")])
    await settle(service)

    state = store.get("a")[1]
    assert (state.status, state.failed_step, state.error) == (
        JobStatus.FAILED,
        Step.TRANSCRIPTION,
        "model error",
    )
    assert status(store, "b") == JobStatus.DONE and len(launcher.launches) == 1


async def test_crash_while_working_fails_only_that_job(started):
    tmp_path, store, service, launcher = started
    launcher.outcomes["a"] = "crash"
    service.enqueue([spec(tmp_path, "a"), spec(tmp_path, "b")])
    await settle(service)

    assert "stopped unexpectedly" in store.get("a")[1].error
    assert status(store, "b") == JobStatus.DONE


async def test_crash_while_loading_fails_the_group_only(started):
    tmp_path, store, service, launcher = started
    launcher.load_crash_models.add("large-v3-turbo")
    service.pause()
    service.enqueue([spec(tmp_path, "a"), spec(tmp_path, "b"), spec(tmp_path, "c", model="small")])
    service.resume()
    await settle(service)

    assert store.get("a")[1].error.startswith("Could not load the model")
    assert status(store, "b") == JobStatus.FAILED
    assert status(store, "c") == JobStatus.DONE


async def test_retry_runs_the_job_again(started):
    tmp_path, store, service, launcher = started
    launcher.outcomes["a"] = "fail"
    service.enqueue([spec(tmp_path, "a")])
    await settle(service)
    launcher.outcomes["a"] = "done"

    service.retry("a")
    await settle(service)

    state = store.get("a")[1]
    assert state.status == JobStatus.DONE and state.error is None


async def test_remove_is_refused_for_the_running_job(started):
    tmp_path, _store, service, launcher = started
    launcher.gates["a"] = asyncio.Event()
    service.enqueue([spec(tmp_path, "a")])
    await until(lambda: launcher.dispatched() == ["a"])
    with pytest.raises(ValueError):
        service.remove("a")
    launcher.gates["a"].set()
    await settle(service)


async def test_status_bug_fails_only_that_job(started):
    tmp_path, store, service, launcher = started
    # JobTranscribed for a job without speaker detection is an invalid transition
    launcher.outcomes["a"] = "done"
    original = FakeHandle.events

    async def events(self):
        async for event in original(self):
            if isinstance(event, JobFileId) and event.job_id == "a":
                yield JobTranscribed("a", 5)
                continue
            if isinstance(event, JobDone) and event.job_id == "a":
                continue
            yield event

    FakeHandle.events = events
    try:
        service.enqueue([spec(tmp_path, "a"), spec(tmp_path, "b")])
        await settle(service)
    finally:
        FakeHandle.events = original

    assert store.get("a")[1].error.startswith("Internal error")
    assert status(store, "b") == JobStatus.DONE


async def test_recovery_after_a_restart(env):
    tmp_path, store, service, _launcher = env
    store.add(
        [
            spec(tmp_path, "a"),
            spec(tmp_path, "b", speaker_detection=True),
            spec(tmp_path, "c", speaker_detection=True),
        ]
    )
    store.update("a", status=JobStatus.TRANSCRIBING, file_id="partial")
    (outputs.TRANSCRIPT_DIR / "partial").mkdir(parents=True)
    for job_id in "bc":
        store.update(job_id, status=JobStatus.TRANSCRIBING)
        store.update(job_id, status=JobStatus.TRANSCRIBED)
        store.update(job_id, status=JobStatus.DIARIZING)
    b_spec = store.get("b")[0]
    outputs.write_checkpoint(
        store.work_dir("b") / RAW_CHECKPOINT,
        transcript={"segments": []},
        audio_duration=5,
        source=b_spec.source,
    )
    service.pause()

    await service.start()
    try:
        assert status(store, "a") == JobStatus.QUEUED
        assert not (outputs.TRANSCRIPT_DIR / "partial").exists()
        assert status(store, "b") == JobStatus.TRANSCRIBED  # valid raw checkpoint
        assert status(store, "c") == JobStatus.QUEUED  # none
    finally:
        await service.stop()


async def test_stop_puts_the_running_job_back(env):
    tmp_path, store, service, launcher = env
    launcher.outcomes["a"] = "hang"
    await service.start()
    service.enqueue([spec(tmp_path, "a")])
    await until(lambda: status(store, "a") == JobStatus.TRANSCRIBING)

    await service.stop()

    assert status(store, "a") == JobStatus.QUEUED
    second = QueueService(store)
    second._lock.acquire()  # the lock was released
    second._lock.release()


HOLD_LOCK = """
import sys, time
from pathlib import Path
from aTrain_core.jobs import QueueLock
lock = QueueLock(Path(sys.argv[1]))
lock.acquire()
print("locked", flush=True)
time.sleep(60)
"""


async def test_start_fails_while_another_process_holds_the_lock(env):
    _tmp_path, store, service, _launcher = env
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_LOCK, str(store.root)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(QueueLockedError):
            await service.start()
    finally:
        holder.kill()
        holder.wait(10)
