from __future__ import annotations
from datetime import datetime, timedelta
import json
from typing import NamedTuple
import cv2
from insightface.app import FaceAnalysis
import numpy as np
import voyager
from packages.face_recognition.algorithm import FaceRecognitionConfig
from src.pipe.function import IO, Function

class FaceRecognitionInput(IO):
    frame: np.ndarray # = np.array([[[0, 0, 0]]])

class FaceResult(NamedTuple):
    name: str # = ""
    image: str # = ""
    distance: float # = 0.0
    box: tuple[int, int, int, int] # = (0, 0, 0, 0)

class BytesOutput(IO):
    data: bytes # = b""

class FaceRecognitionFunction(Function):
    cls_input: type[FaceRecognitionInput] = FaceRecognitionInput

    def __init__(self, config: FaceRecognitionConfig) -> None:
        self.config: FaceRecognitionConfig = config
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=self.config.ctx_id, det_thresh=self.config.det_thresh, det_size=self.config.det_size)
        if self.config.data:
            self.names: list[tuple[str, str]] = self.config.data.names
            self.embeddings: voyager.Index = voyager.Index(voyager.Space.Euclidean, num_dimensions=512)
            self.embeddings.add_items(self.config.data.embeddings)
            self.no_data: bool = False
        else:
            self.no_data: bool = True
        self.latest: dict[str, datetime] = {}
        self.frame_output = self.config.frame_output.cls_function(self.config.frame_output.config) if self.config.frame_output else None
        self.results_output = self.config.results_output.cls_function(self.config.results_output.config) if self.config.results_output else None
        self.attendance_output = self.config.attendance_output.cls_function(self.config.attendance_output.config) if self.config.attendance_output else None

    def __call__(self, input: FaceRecognitionInput) -> IO:
        faces = self.app.get(input.frame)
        res: list[FaceResult] = []
        if self.no_data:
            for face in faces:
                res.append(FaceResult(
                    name=self.config.unknown,
                    image="",
                    distance=0.0,
                    box=list(map(int, face.bbox)),
                ))
        else:
            for face in faces:
                neighbors, distances = self.embeddings.query(face.normed_embedding, k=1)
                res.append(FaceResult(
                    name=self.names[neighbors[0]][0] if distances[0] < self.config.threshold else self.config.unknown,
                    image=self.names[neighbors[0]][1],
                    distance=float(distances[0]),
                    box=list(map(int, face.bbox)),
                ))
        if self.frame_output:
            if self.config.annotate:
                padding = self.config.annotations.padding
                half_padding = padding // 2
                for face in res:
                    face_box = face.box
                    (text_width, text_height) = cv2.getTextSize(face.name, self.config.annotations.font, self.config.annotations.text_scale, self.config.annotations.text_thickness)[0]
                    cv2.rectangle(input.frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), 
                        self.config.annotations.known_box_color if face.name != self.config.unknown else self.config.annotations.unknown_box_color, self.config.annotations.box_thickness)
                    cv2.rectangle(input.frame, (face_box[0], face_box[1] - padding - text_height), (face_box[0] + padding + text_width, face_box[1]), (255, 255, 255), cv2.FILLED)
                    cv2.putText(input.frame, face.name, (face_box[0] + half_padding, face_box[1] - half_padding), self.config.annotations.font, self.config.annotations.text_scale, (0, 0, 0), self.config.annotations.text_thickness)

            flag, enc = cv2.imencode('.jpg', input.frame)
            if flag:
                self.frame_output(BytesOutput(
                    data=enc.tobytes(),
                ))
            # if flag:
            #     self.channels["frame"].update((b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + enc.tobytes() + b'\r\n'))
            # if self.config["frame"]["file"]:
            #     self.videowriter.write(img)

        if res:
            if self.results_output:
                self.results_output(BytesOutput(
                    data=(json.dumps({str(datetime.now()): [face._asdict() for face in res]}) + "\n").encode("latin-1"),
                ))

            if not self.no_data and self.attendance_output:
                names: list[str] = []
                now: datetime = datetime.now()
                for face in res:
                    name: str = face.name
                    if (name != self.config.unknown):
                        if (name not in self.latest) or (now - self.latest[name] > timedelta(seconds=self.config.attendance_interval)):
                            names.append(name)
                        self.latest[name] = now

                if names:
                    # self.attendance_output("".join(f"{now},{name}\n" for name in names).encode("latin-1"))
                    self.attendance_output(BytesOutput(
                        data="".join(["{\"", str(now), "\": ",  str(names).replace("'", '"'), "}\n"]).encode("latin-1"),
                    ))
        return IO()
