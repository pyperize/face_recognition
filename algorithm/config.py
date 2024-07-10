from __future__ import annotations
import flet as ft
import cv2
from src.pipe import Pipe, Config, ConfigUI
from src.ui.pipe.tile import PipeTile
from typing import Any, Callable, NamedTuple, TYPE_CHECKING
from packages.face_recognition.data import FaceRecognitionDataOutput
if TYPE_CHECKING:
    from src.manager import Manager
    from src.ui.common import ConfigPage
    from packages.face_recognition.algorithm import FaceRecognitionPipe

class AnnotationConfig(NamedTuple):
    padding: int = 20
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_scale: float = 0.5
    text_thickness: int = 1
    known_box_color: tuple[int, int, int] = (0, 255, 0)
    unknown_box_color: tuple[int, int, int] = (0, 0, 255)
    box_thickness: int = 2

class FaceRecognitionConfig(Config):
    frame_output: Pipe | None = None
    results_output: Pipe | None = None
    attendance_output: Pipe | None = None
    data: FaceRecognitionDataOutput | None = None
    ctx_id: int = 0
    det_thresh: float = 0.5
    det_size: tuple[int, int] = (640, 640)
    threshold: float = 1.0
    unknown: str = "Unknown"
    attendance_interval: int = 300
    annotate: bool = True
    annotations: AnnotationConfig = AnnotationConfig()

class FaceRecognitionConfigUI(ConfigUI):
    def __init__(self, instance: FaceRecognitionPipe, manager: Manager, config_page: ConfigPage) -> None:
        super().__init__(instance, manager, config_page)
        self.instance: FaceRecognitionPipe = instance
        self.data_options: list[ft.dropdown.Option] = []
        self.refresh_data_options(False)
        self.content: ft.Column = ft.Column([
            PipeTile(
                "Frame",
                self.manager,
                self.config_page,
                self.select_pipe,
                self.delete_pipe(0),
                self.instance.config.frame_output,
            ),
            PipeTile(
                "Results",
                self.manager,
                self.config_page,
                self.select_pipe,
                self.delete_pipe(1),
                self.instance.config.results_output,
            ),
            PipeTile(
                "Attendance",
                self.manager,
                self.config_page,
                self.select_pipe,
                self.delete_pipe(2),
                self.instance.config.attendance_output,
            ),
            ft.Dropdown(
                self.instance.config.data.name if self.instance.config.data else None,
                label="Data source to use",
                hint_text="Data source to use",
                options=self.data_options,
                on_click=self.refresh_data_options,
                border_color="grey",
                dense=True,
            ),
            ft.TextField(str(self.instance.config.threshold), label="Distance Threshold", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
            ft.TextField(str(self.instance.config.det_size[1]), label="Detection Width", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.det_size[0]), label="Detection Height", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.det_thresh), label="Detection Threshold", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
            ft.TextField(self.instance.config.unknown, label="Label for Unknowns", border_color="grey"),
            ft.TextField(str(self.instance.config.attendance_interval), label="Interval after Latest Detection", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.Switch(label="Annotate Output Frames", value=self.instance.config.annotate),
            ft.Container(
                ft.Text("Annotation Configuration"),
                padding=ft.padding.symmetric(vertical=20),
            ),
            ft.TextField("%02x%02x%02x" % self.instance.config.annotations.known_box_color, label="Known Box Color (hex)", border_color="grey"),
            ft.TextField("%02x%02x%02x" % self.instance.config.annotations.unknown_box_color, label="Unknown Box Color (hex)", border_color="grey"),
            ft.TextField(str(self.instance.config.annotations.box_thickness), label="Box Thickness", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.annotations.text_scale), label="Text Scale", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
            ft.TextField(str(self.instance.config.annotations.text_thickness), label="Text Thickness", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.annotations.padding), label="Padding", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
        ])

    def refresh_data_options(self, update: bool = True):
        self.data_options.clear()
        self.data_options.extend([ft.dropdown.Option(id) for id in self.manager.data])
        if update:
            self.update()

    def select_pipe(self, cls: type[Pipe] | Pipe) -> Pipe:
        if isinstance(cls, type):
            cls: Pipe = cls(cls.cls_name, self.manager, cls.cls_config())
        return cls

    def delete_pipe(self, index: int) -> Callable[[Any], None]:
        def _delete_pipe(e) -> None:
            self.content.controls[index].pipe_selector.value = None
            self.content.controls[index].select_changed(None)
            self.update()
        return _delete_pipe

    def dismiss(self) -> None:
        self.instance.config = FaceRecognitionConfig(
            frame_output=self.content.controls[0].instance,
            results_output=self.content.controls[1].instance,
            attendance_output=self.content.controls[2].instance,
            data=self.manager.data[self.content.controls[3].value] if self.content.controls[3].value else None,
            threshold=float(self.content.controls[4].value),
            det_size=(int(self.content.controls[5].value), int(self.content.controls[6].value)),
            det_thresh=float(self.content.controls[7].value),
            unknown=self.content.controls[8].value,
            attendance_interval=int(self.content.controls[9].value),
            annotate=self.content.controls[10].value,
            annotations=AnnotationConfig(
                known_box_color=tuple(bytes.fromhex(self.content.controls[12].value)),
                unknown_box_color=tuple(bytes.fromhex(self.content.controls[13].value)),
                box_thickness=int(self.content.controls[14].value),
                text_scale=float(self.content.controls[15].value),
                text_thickness=int(self.content.controls[16].value),
                padding=int(self.content.controls[17].value),
            ),
        )
