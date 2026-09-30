from collections.abc import Callable
from contextlib import contextmanager

from aTrain.components.layout.footer import footer
from aTrain.components.layout.header import header
from aTrain.components.layout.sidebar import sidebar
from nicegui import ui


@contextmanager
def base_layout(below: Callable[[], None] | None = None):
    """The page frame. `below` fills a second box under the main one."""
    ui.query("body").classes("bg-gray-100")
    drawer_handle = sidebar()
    header(drawer_handle)
    with ui.card().classes("w-full h-full bg-white rounded-lg p-8").props("flat"):
        yield
    if below is not None:
        with ui.card().classes("w-full bg-white rounded-lg p-8").props("flat"):
            below()
    footer()
