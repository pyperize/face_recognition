from __future__ import annotations
from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
from src.pipe.function import IO, Function
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from packages.face_recognition.data import FaceRecognitionDataConfig

class FaceRecognitionDataOutput(IO):
    name: str = "Face Recognition Data"
    names: list[tuple[str, str]] = []
    embeddings: np.ndarray = np.array([])

class FaceRecognitionDataFunction(Function):
    cls_output: type[FaceRecognitionDataOutput] = FaceRecognitionDataOutput

    def __init__(self, config: FaceRecognitionDataConfig) -> None:
        self.config: FaceRecognitionDataConfig = config

    def get_biggest_face(self, faces: list) -> int:
        biggest: int = 0
        max_area: float = (faces[0].bbox[2] - faces[0].bbox[0]) * (faces[0].bbox[3] - faces[0].bbox[1])
        for count in range(1, len(faces)):
            face = faces[count]
            area: float = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            if area > max_area:
                max_area = area
                biggest = count
        return biggest

    def __call__(self, input: IO = IO()) -> FaceRecognitionDataOutput:
        app: FaceAnalysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=self.config.ctx_id, det_thresh=self.config.det_thresh, det_size=self.config.det_size)

        names: list[tuple[str, str]] = []
        embeddings: list[np.ndarray] = []

        for dirpath, dirnames, filenames in tqdm(os.walk(self.config.db_path)):
            for filename in filenames:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    img_name: str = os.path.join(dirpath, filename)
                    faces: list = app.get(np.array(Image.open(img_name))[:, :, ::-1])
                    if len(faces) == 0: raise Exception(f"{img_name} has 0 faces detected")
                    embeddings.append(faces[self.get_biggest_face(faces)].normed_embedding)
                    names.append((os.path.basename(os.path.normpath(dirpath)).replace(".", "/"), filename))

        return FaceRecognitionDataOutput(
            name=self.config.name,
            names=names,
            embeddings=np.array(embeddings),
        )
