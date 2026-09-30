import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast
from uuid import uuid4

from aTrain.components.dialogs.error import dialog_error
from aTrain.components.dialogs.finished import dialog_finished
from aTrain.components.dialogs.process import close_dialog_process, dialog_process
from aTrain_core.settings import Device, check_inputs_transcribe
from nicegui import app, events, ui


class State(TypedDict):
    model: str
    language: str
    speaker_detection: bool
    speaker_count: float | None
    GPU: bool
    compute_type: str
    temperature_override: float | None
    initial_prompt: str | None
    cpu_threads: int


@dataclass(slots=True)
class UploadPayload:
    """Platform-neutral description of the file to transcribe.

    Two callers feed the same pipeline: the browser upload on Windows/macOS
    (a NiceGUI `UploadEventArguments`) and the native file picker on
    Linux/Flatpak (a path on disk). Normalising both into this type keeps
    NiceGUI's event class out of the transcription code, so an upload-API
    change like 2.x `.name`/`.content` -> 3.x `.file` can only ever break the
    one adapter below instead of silently breaking one of the two platforms.
    """

    name: str
    upload: ui.upload.FileUpload | None = None
    path: Path | None = None

    async def materialise(self, directory: Path, filename: str) -> Path:
        """Return a path the engine can read, staging the upload if needed."""
        if self.path is not None:
            return self.path  # already on disk - no copy, the picker gave us a real file
        if self.upload is None:
            raise ValueError(f"No file to transcribe: {self.name!r} has neither upload nor path")
        target = directory / filename
        await self.upload.save(target)
        return target


async def start_transcription(file: events.UploadEventArguments):
    """NiceGUI `on_upload` handler (browser upload path)."""
    await run_pipeline(UploadPayload(name=file.file.name, upload=file.file))


async def start_transcription_from_path(path: Path, name: str):
    """Entry point for the native file picker (Linux/Flatpak path)."""
    await run_pipeline(UploadPayload(name=name, path=path))


async def run_pipeline(payload: UploadPayload):
    """Add the file as one job to the queue and follow it in the progress dialog."""
    # Lazy import for improved startup speed
    from aTrain.utils.queue_ui import LOCKED_TEXT, build_spec_from_state, get_queue_service
    from aTrain_core.jobs import QueueLockedError
    from werkzeug.utils import secure_filename

    job_id = uuid4().hex
    progress = {"task": "Prepare", "current": 0, "total": 999999}
    # Snapshot rather than read live: the page re-renders while the upload is
    # staged, and `get_model_options` resets `model` to None whenever no model
    # is on disk yet. Reading through to the live storage would also let a user
    # switching the model mid-upload retarget a run that is already under way.
    state = cast(State, dict(app.storage.general))
    try:
        device = Device.GPU if state.get("GPU") else Device.CPU
        # Validate first: it needs nothing but the name and the settings, and
        # rejecting a wrong model or language should not cost a full copy of the
        # upload beforehand.
        check_inputs_transcribe(payload.name, state.get("model"), state.get("language"), device)
        service = await get_queue_service()
    except QueueLockedError:
        ui.notify(LOCKED_TEXT, color="negative", multi_line=True)
        return
    except Exception as e:
        dialog_error(error=str(e), traceback=traceback.format_exc())
        return

    dialog_process(progress, on_stop=lambda: service.cancel([job_id]))
    try:
        # A browser upload is staged where the queue deletes it once the job is done.
        staging = service.store.uploads_root / job_id
        staging.mkdir(parents=True, exist_ok=True)
        source = await payload.materialise(staging, secure_filename(payload.name) or "upload")
        if payload.path is not None:
            staging.rmdir()
        spec = build_spec_from_state(state, job_id=job_id, source=source, display_name=payload.name)
        follow_job(service, job_id, progress)
        service.enqueue([spec])
    except Exception as e:
        close_dialog_process()
        dialog_error(error=str(e), traceback=traceback.format_exc())


def follow_job(service, job_id: str, progress: dict) -> None:
    """Feed the progress dialog from the queue, then show how the job ended."""
    from aTrain_core.jobs import FINAL_STATUSES, JobStatus

    client = ui.context.client

    def on_change(changed_id: str) -> None:
        try:
            _, state = service.store.get(job_id)
        except KeyError:
            unsubscribe()
            return
        if state.status == JobStatus.QUEUED:
            ahead = jobs_ahead(service, job_id)
            progress["task"] = f"Waiting: {ahead} jobs ahead" if ahead else "Prepare"
        elif changed_id == job_id and (event := service.progress(job_id)) is not None:
            progress.update(task=event.task, current=event.current, total=event.total)
        if state.status not in FINAL_STATUSES:
            return
        unsubscribe()
        with client:
            close_dialog_process()
            if state.status == JobStatus.DONE:
                dialog_finished(state.file_id)
            elif state.status == JobStatus.FAILED:
                dialog_error(error=state.error or "", traceback=state.traceback or "")
            else:
                ui.navigate.reload()
        # Nothing shows finished jobs yet (the Queue tab comes later), so don't keep them.
        service.remove(job_id)

    unsubscribe = service.subscribe(on_change)


def jobs_ahead(service, job_id: str) -> int:
    from aTrain_core.jobs import FINAL_STATUSES

    ahead = 0
    for spec, state in service.jobs():
        if spec.id == job_id:
            break
        ahead += state.status not in FINAL_STATUSES
    return ahead
