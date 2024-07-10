from __future__ import annotations
import flet as ft
from src.pipe.config import Config, ConfigUI
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.manager import Manager
    from src.ui.common import ConfigPage
    from packages.face_recognition.data import FaceRecognitionDataPipe

class FaceRecognitionDataConfig(Config):
    name: str = "Face Recognition Data"
    ctx_id: int = 0
    det_thresh: float = 0.5
    det_size: tuple[int, int] = (640, 640)
    db_path: str = "./../../db"

class FaceRecognitionDataConfigUI(ConfigUI):
    def __init__(self, instance: FaceRecognitionDataPipe, manager: Manager, config_page: ConfigPage, content: ft.Control | None = None) -> None:
        self.instance: FaceRecognitionDataPipe = instance
        super().__init__(instance, manager, config_page, ft.Column(spacing=20, controls=[
            ft.TextField(self.instance.config.name, label="Unique Database Name", border_color="grey"),
            ft.TextField(self.instance.config.db_path, label="Database Path", border_color="grey"),
            ft.TextField(self.instance.config.det_size[0], label="Detection Width", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(self.instance.config.det_size[1], label="Detection Height", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(self.instance.config.det_thresh, label="Detection Threshold", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
        ]))

    def dismiss(self) -> None:
        self.instance.config = FaceRecognitionDataConfig(
            name=self.content.controls[0].value,
            db_path=self.content.controls[1].value,
            det_size=(int(self.content.controls[3].value), int(self.content.controls[2].value)),
            det_thresh=float(self.content.controls[4].value),
        )
