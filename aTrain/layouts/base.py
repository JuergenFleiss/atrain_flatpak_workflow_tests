from contextlib import contextmanager

from aTrain.components.layout.footer import footer
from aTrain.components.layout.header import header
from aTrain.components.layout.sidebar import sidebar
from nicegui import ui


@contextmanager
def base_layout():
    ui.query("body").classes("bg-gray-100")
    drawer_handle = sidebar()
    header(drawer_handle)
    with ui.card().classes("w-full h-full bg-white rounded-lg p-8").props("flat"):
        yield
    footer()
