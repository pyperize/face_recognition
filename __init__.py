from __future__ import annotations
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from packages.face_recognition.algorithm import FaceRecognitionPipe
from packages.face_recognition.data import FaceRecognitionDataPipe
from src.package.package import Package
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from src.pipe import Pipe

class FaceRecognitionPackage(Package):
    name: str = "Face Recognition"
    _pipes: Iterable[type[Pipe]] = [FaceRecognitionPipe, FaceRecognitionDataPipe]
    dependencies: dict[str, Package] = {}
