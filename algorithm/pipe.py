from __future__ import annotations
import src.pipe as pipe
import packages.face_recognition.algorithm as face_recognition
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.manager import Manager
    from src.ui.common import ConfigPage

class FaceRecognitionPipe(pipe.Pipe):
    cls_name: str = "Face Recognition"
    cls_config: type[face_recognition.FaceRecognitionConfig] = face_recognition.FaceRecognitionConfig
    cls_function: type[face_recognition.FaceRecognitionFunction] = face_recognition.FaceRecognitionFunction

    def __init__(self, name: str, manager: Manager, config: face_recognition.FaceRecognitionConfig | None = None) -> None:
        super().__init__(name, manager, config)
        self.config: face_recognition.FaceRecognitionConfig = config if config else self.cls_config()

    def config_ui(self, manager: Manager, config_page: ConfigPage) -> face_recognition.FaceRecognitionConfigUI:
        return face_recognition.FaceRecognitionConfigUI(self, manager, config_page)

    def play(self, manager: Manager) -> None:
        if self.playing:
            return
        self.playing = True
        if self.config.frame_output: self.config.frame_output.play(manager)
        if self.config.results_output: self.config.results_output.play(manager)
        if self.config.attendance_output: self.config.attendance_output.play(manager)

    def stop(self, manager: Manager, result: pipe.IO) -> None:
        if not self.playing:
            return
        self.playing = False
        if self.config.frame_output: self.config.frame_output.stop(manager, result)
        if self.config.results_output: self.config.results_output.stop(manager, result)
        if self.config.attendance_output: self.config.attendance_output.stop(manager, result)
