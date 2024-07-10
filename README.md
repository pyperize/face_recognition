# Face Recognition pipe for pyperize
Using InsightFace and Spotify Voyager

## Prerequisites
- CUDA 12

## Install

1. Copy this package into ```./packages/```
2. Edit ```./packages/__init__.py``` to import the package
3. Add the package name and instance to the ```PACKAGES``` global variable in ```./packages/__init__.py```

```./packages/__init__.py``` should contain something like this where ```...``` are the other packages

```
from src.package import Package
from packages import (
    ...
    face_recognition,
    ...
)

PACKAGES: dict[str, Package] = {
    ...
    face_recognition.FaceRecognitionPackage.name: face_recognition.FaceRecognitionPackage(),
    ...
}
```
