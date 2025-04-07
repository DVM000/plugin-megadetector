# Animal detection from camera-trap images using MegaDetector

This app integrates [MegaDectector](https://github.com/microsoft/CameraTraps/) into SAGE nodes for animal detection.


## How to Use
To run the program,

```bash
# Captures and publishes detections and images containing animals 
python3 main.py --stream bottom_camera --n 10 --threshold 0.2
```

this will capture 10 images, run MegaDetector on each image, publish animal detections on topic `env.megadetector.animal` (value for a topic indicates the confidence of the detection) and upload images containing animals. If employing `--annotations` argument, uploaded images will be annotated with bounding boxes.


