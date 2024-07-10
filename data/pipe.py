from __future__ import annotations
import src.pipe as pipe
import packages.face_recognition.data as face_recognition
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.manager import Manager
    from src.ui.common import ConfigPage

class FaceRecognitionDataPipe(pipe.Pipe):
    cls_name: str = "Face Recognition (Data)"
    cls_config: type[pipe.Config] = face_recognition.FaceRecognitionDataConfig
    cls_function: type[pipe.Function] = face_recognition.FaceRecognitionDataFunction

    def __init__(self, name: str, manager: Manager, config: face_recognition.FaceRecognitionDataConfig) -> None:
        super().__init__(name, manager, config)
        self.config: face_recognition.FaceRecognitionDataConfig = config

    def config_ui(self, manager: Manager, config_page: ConfigPage) -> face_recognition.FaceRecognitionDataConfigUI:
        return face_recognition.FaceRecognitionDataConfigUI(self, manager, config_page)

    def play(self, manager: Manager) -> None:
        if self.playing:
            return
        self.playing = True

    def stop(self, manager: Manager, result: face_recognition.FaceRecognitionDataOutput) -> None:
        if not self.playing:
            return
        self.playing = False
        if result:
            manager.data[self.config.name] = result
