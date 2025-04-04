# Animal detection from camera-trap images using MegaDetector

This app integrates [MegaDectector](https://github.com/microsoft/CameraTraps/) into SAGE nodes for animal detection.

## What is MegaDetector?

Megadetector is an open-source object detection model developed by Microsoft AI for Earth. It is trained to detect three types of objects in camera trap images:
animal, human and vehicle. MegaDetector is optimized for ecological and conservation use cases, helping automate wildlife monitoring by identifying 
and classifying entities in images from camera traps

## How to Use
To run the programÑ

```bash
# Captures and publishes detections and images containing animals 
python3 main.py --stream bottom_camera --n 10 --threshold 0.2
```

this will capture 10 images, run MegaDetector on each image, publish animal detections on topic `env.megadetector.animal` (value for a topic indicates the confidence of the detection) and upload images containing animals. If employing `--annotations` argument, uploaded images will be annotated with bounding boxes.

