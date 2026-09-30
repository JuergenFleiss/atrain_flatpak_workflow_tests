"""The Queue tab and several-file input, with a real but paused QueueService on a temporary
store (it never starts a child)."""

import asyncio

import aTrain_core.transcribe  # noqa: F401  pre-import so the splash import is instant
import pytest
from aTrain.components.settings.file import folder_summary
from aTrain.utils import queue_ui
from aTrain.utils.transcription import start_paths
from aTrain_core.jobs import JobStatus, JobStore, QueueLockedError
from aTrain_core.queue_service import QueueService
from aTrain_core.settings import ComputeType, Device
from nicegui import app, ui
from nicegui.testing import User
from tests.unit.test_jobs import make_spec

CHEAP_SETTINGS = {
    "model": "tiny",
    "language": "auto-detect",
    "speaker_detection": False,
    "speaker_count": 0,
    "GPU": False,
    "compute_type": "int8",
    "temperature_override": None,
    "initial_prompt": None,
    "cpu_threads": 0,
}


def no_child(arg):
    raise AssertionError("the UI tests never start a phase child")


@pytest.fixture
async def service(tmp_path, monkeypatch):
    service = QueueService(
        JobStore(tmp_path / "queue"), launch_phase1=no_child, launch_phase2=no_child
    )
    service.pause()
    await service.start()

    async def get_service():
        return service

    monkeypatch.setattr(queue_ui, "get_queue_service", get_service)
    yield service
    await service.stop()


def add(service, tmp_path, job_id, *statuses, **overrides):
    source = tmp_path / f"{job_id}.mp3"
    source.write_bytes(b"audio")
    values = {"device": Device.CPU, "compute_type": ComputeType.INT8, "speaker_detection": False}
    values.update(overrides)
    service.enqueue(
        [
            make_spec(
                id=job_id, source=source, display_name=f"{job_id}.mp3", export_dir=None, **values
            )
        ]
    )
    for status in statuses:
        service.store.update(job_id, status=status)


async def until(condition):
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached")


S = JobStatus


async def test_every_status_is_shown(service, tmp_path, user: User):
    add(service, tmp_path, "queued")
    add(service, tmp_path, "running", S.TRANSCRIBING)
    add(service, tmp_path, "waiting", S.TRANSCRIBING, S.TRANSCRIBED, speaker_detection=True)
    add(
        service,
        tmp_path,
        "speakers",
        S.TRANSCRIBING,
        S.TRANSCRIBED,
        S.DIARIZING,
        speaker_detection=True,
    )
    add(service, tmp_path, "done", S.TRANSCRIBING, S.DONE)
    add(service, tmp_path, "failed", S.FAILED)
    add(service, tmp_path, "cancelled", S.CANCELLED)

    await user.open("/queue")

    for text in (
        "Queued",
        "Transcribing",
        "Waiting for speaker detection",
        "Detecting speakers",
        "Done",
        "Failed ⓘ",
        "Cancelled",
        "cancelled.mp3",
        "large-v3-turbo · de · no speakers · CPU",
    ):
        await user.should_see(text)
    await user.should_see("Queue")  # the sidebar entry


async def test_remove_asks_first(service, tmp_path, user: User):
    add(service, tmp_path, "a")
    await user.open("/queue")

    user.find(kind=ui.button, content="✕").click()
    await user.should_see("Remove a.mp3 from the queue?")
    user.find(marker="confirm_ok").click()

    await until(lambda: service.jobs() == [])


async def test_running_job_is_cancelled_not_removed(service, tmp_path, user: User):
    add(service, tmp_path, "a", S.TRANSCRIBING)
    await user.open("/queue")

    user.find(kind=ui.button, content="✕").click()
    await user.should_see("Stop transcribing a.mp3?")
    user.find(marker="confirm_ok").click()

    await until(lambda: service.store.get("a")[1].status == S.CANCELLED)


async def test_retry(service, tmp_path, user: User):
    add(service, tmp_path, "b", S.FAILED)
    await user.open("/queue")

    user.find(kind=ui.button, content="retry").click()

    await until(lambda: service.store.get("b")[1].status == S.QUEUED)
    await user.should_see("↓")  # a waiting job can be moved


async def test_clear_finished_keeps_the_others(service, tmp_path, user: User):
    add(service, tmp_path, "a")
    add(service, tmp_path, "b", S.TRANSCRIBING, S.DONE)
    add(service, tmp_path, "c", S.CANCELLED)
    await user.open("/queue")

    user.find(kind=ui.button, content="Clear finished").click()
    await user.should_see("Remove 2 finished, failed or cancelled jobs from the queue?")
    user.find(marker="confirm_ok").click()

    await until(lambda: [spec.id for spec, _ in service.jobs()] == ["a"])


async def test_pause_and_resume(service, user: User):
    await user.open("/queue")
    user.find(kind=ui.button, content="Resume").click()
    await until(lambda: not service.paused)
    await user.should_see("Pause")


async def test_locked_queue_shows_a_banner(monkeypatch, user: User):
    async def locked():
        raise QueueLockedError("held")

    monkeypatch.setattr(queue_ui, "get_queue_service", locked)
    await user.open("/queue")
    await user.should_see(queue_ui.LOCKED_TEXT)


async def test_several_files_go_to_the_queue(service, tmp_path, user: User):
    await user.open("/queue")
    app.storage.general.update(CHEAP_SETTINGS)
    paths = []
    for name in ("one.mp3", "two.mp3"):
        (tmp_path / name).write_bytes(b"audio")
        paths.append(tmp_path / name)

    with user:
        await start_paths(paths, export_dir=tmp_path / "transcriptions")

    jobs = service.jobs()
    assert [spec.display_name for spec, _ in jobs] == ["one.mp3", "two.mp3"]
    assert all(spec.export_dir == tmp_path / "transcriptions" for spec, _ in jobs)
    await user.should_see("one.mp3")


def test_folder_summary(tmp_path):
    for name in ("a.mp3", "b.mp3", "c.mp3", "d.mp3", "e.mp3", "f.mp3", "notes.txt", ".hidden"):
        (tmp_path / name).write_text("x")
    files = sorted(tmp_path.glob("*.mp3"))

    lines = folder_summary(tmp_path, files).splitlines()

    assert lines[0] == "6 supported files found (1 other files ignored)"
    assert lines[1:] == ["a.mp3", "b.mp3", "c.mp3", "d.mp3", "e.mp3", "and 1 more"]


async def test_transcribe_page_shows_the_queue_below_the_settings(service, user: User):
    await user.open("/")
    await user.should_see("Advanced Settings", retries=200)
    await user.should_see("The queue is empty.")


async def test_one_file_goes_to_the_queue_without_a_dialog(service, tmp_path, user: User):
    await user.open("/queue")
    app.storage.general.update(CHEAP_SETTINGS)
    (tmp_path / "one.mp3").write_bytes(b"audio")

    with user:
        await start_paths([tmp_path / "one.mp3"])

    assert [spec.display_name for spec, _ in service.jobs()] == ["one.mp3"]
    await user.should_see("one.mp3 added to the queue")
    user.should_not_see(kind=ui.dialog)
