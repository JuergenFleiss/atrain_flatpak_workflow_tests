"""Unit tests for aTrain_core.jobs.JobStore (queue.json persistence)."""

import json

import pytest
from aTrain_core.jobs import InvalidTransitionError, JobStatus, JobStore
from tests.unit.test_jobs import make_spec


def ids(store: JobStore) -> list[str]:
    return [spec.id for spec, _ in store.jobs()]


def test_add_and_reload(tmp_path):
    store = JobStore(tmp_path)
    store.add([make_spec(id="a"), make_spec(id="b", speaker_detection=False)])
    store.update("a", status=JobStatus.TRANSCRIBING, started_at="2026-09-30 14-06-40")

    reloaded = JobStore(tmp_path)
    assert ids(reloaded) == ["a", "b"]
    assert reloaded.get("a") == store.get("a")
    assert not (tmp_path / "queue.json.tmp").exists()


def test_add_rejects_duplicate_ids(tmp_path):
    store = JobStore(tmp_path)
    store.add([make_spec(id="a")])
    with pytest.raises(ValueError):
        store.add([make_spec(id="a")])


def test_update_checks_transitions(tmp_path):
    store = JobStore(tmp_path)
    store.add([make_spec(id="a", speaker_detection=False)])
    with pytest.raises(InvalidTransitionError):
        store.update("a", status=JobStatus.DONE)
    store.update("a", status=JobStatus.TRANSCRIBING)
    with pytest.raises(InvalidTransitionError):
        store.update("a", status=JobStatus.TRANSCRIBED)  # no speaker detection
    assert store.update("a", status=JobStatus.DONE).status == JobStatus.DONE


def test_move(tmp_path):
    store = JobStore(tmp_path)
    store.add([make_spec(id=job_id) for job_id in "abc"])
    store.move("c", -1)
    assert ids(store) == ["a", "c", "b"]
    store.move("a", -1)  # already first
    store.move("a", 5)  # clamped to the end
    assert ids(JobStore(tmp_path)) == ["c", "b", "a"]


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"schema_version": 99, "jobs": []}),
        json.dumps({"schema_version": 1}),
    ],
)
def test_unreadable_queue_is_moved_aside(tmp_path, content):
    (tmp_path / "queue.json").write_text(content, encoding="utf-8")
    store = JobStore(tmp_path)
    assert store.jobs() == []
    assert [p.name.startswith("queue.json.unreadable-") for p in tmp_path.iterdir()] == [True]


def test_remove_deletes_work_folder_and_staged_upload_only(tmp_path):
    outside = tmp_path / "Interviews" / "interview.mp3"
    outside.parent.mkdir()
    outside.write_text("audio")
    store = JobStore(tmp_path / "queue")
    staged = store.uploads_root / "a" / "upload.mp3"
    staged.parent.mkdir(parents=True)
    staged.write_text("audio")
    store.add([make_spec(id="a", source=staged), make_spec(id="b", source=outside)])
    for job_id in "ab":
        store.work_dir(job_id).mkdir(parents=True)

    store.remove("a")
    store.remove("b")

    assert store.jobs() == []
    assert not staged.parent.exists() and not store.work_dir("a").exists()
    assert not store.work_dir("b").exists()
    assert outside.exists()


def test_clear_finished_keeps_active_jobs(tmp_path):
    store = JobStore(tmp_path)
    store.add([make_spec(id=job_id, speaker_detection=False) for job_id in "abcd"])
    for job_id in "abc":
        store.update(job_id, status=JobStatus.TRANSCRIBING)
    store.update("a", status=JobStatus.DONE)
    store.update("b", status=JobStatus.FAILED)

    assert store.clear_finished() == ["a", "b"]
    assert ids(store) == ["c", "d"]
