# Student Engagement Detection — Demo

Real-time student engagement detection using YOLOv8 face detection and ResNet50V2 classification.

## What it does

Opens your webcam, detects your face using YOLOv8, and classifies whether you are engaged or not engaged in real time. The bounding box colour transitions from red (not engaged) to amber to green (engaged) based on the model's confidence.

## Requirements

- Python 3.11
- Windows (tested on Windows 11 with RTX 3060)

## Setup
```bash
python -m venv demo-env
demo-env\Scripts\activate
pip install tensorflow-cpu==2.10.0 tensorflow-directml-plugin opencv-python ultralytics numpy
```

## Files needed

- `demo.py` — main script
- `config.py` — paths and settings
- `yolov8n-face-lindevs.pt` — YOLOv8 face detection model
- `model.keras` — trained ResNet50V2 engagement classifier (not included, download separately)

## Usage

Update `MODEL_PATH` in `config.py` to point to your downloaded model file, then run:
```bash
python demo.py
```

Press `Q` to quit.

## Model

The engagement classifier is a ResNet50V2 model trained on the DAiSEE dataset using two-stage transfer learning. It classifies each face crop as either Engaged or Not Engaged.

The full training code is available in the main repository.

## Notes

- YOLOv8 runs on CPU on Windows
- TensorFlow uses DirectML for GPU acceleration if available
- Detection runs every 5 frames to maintain smooth display FPS