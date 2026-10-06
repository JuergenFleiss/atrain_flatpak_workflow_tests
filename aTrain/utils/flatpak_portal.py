"""Flatpak only: files outside the sandbox, through the desktop portals.

A Flatpak can read only the files the user gives it through a portal: dropped files through
the file transfer portal, other files through the file chooser. Light to import, since the
window process (aTrain.utils.linux_drop) uses it too.
"""

from pathlib import Path

# A drag offers its files to sandboxed apps under one of these types, GTK 4 apps (e.g. GNOME
# Files) under the first, older GTK under the second. The data is a key for RetrieveFiles.
TRANSFER_TARGETS = ("application/vnd.portal.filetransfer", "application/vnd.portal.files")


def transfer_target(offered: list[str]) -> str | None:
    """The file transfer type a drag offers, if any."""
    return next((target for target in TRANSFER_TARGETS if target in offered), None)


def transfer_key(data: bytes) -> str:
    return data.decode("utf-8", "replace").strip("\0 \r\n")


def retrieve_files(key: str) -> list[str]:
    """The files of a file transfer, as paths in the document portal that the sandbox can
    read."""
    from gi.repository import Gio, GLib

    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    result = bus.call_sync(
        "org.freedesktop.portal.Documents",
        "/org/freedesktop/portal/documents",
        "org.freedesktop.portal.FileTransfer",
        "RetrieveFiles",
        GLib.Variant("(sa{sv})", (key, {})),
        GLib.VariantType("(as)"),
        Gio.DBusCallFlags.NONE,
        10_000,
        None,
    )
    return list(result.unpack()[0])


async def choose_dropped(dropped: Path) -> tuple[list[str], bool]:
    """A dropped path the sandbox can't read (dragged from an app that doesn't offer the file
    transfer portal): the portal's file chooser opens where it is, for one more click.
    Returns the picked paths and whether a folder was asked for (no extension)."""
    from aTrain.utils.file_selection import pick_native  # it imports this module
    from nicegui import run, ui

    folder = not dropped.suffix
    kind = "the folder " if folder else ""
    ui.notify(f"Select {kind}{dropped.name} to give aTrain access to it.", multi_line=True)
    return await run.io_bound(pick_native, folder, dropped.parent), folder
