#https://github.com/microsoft/CameraTraps/blob/main/demo/image_separation_demo.py

from datetime import datetime, timezone
import argparse
import os
import logging
import time
import shutil
import torch

from waggle.plugin import Plugin
from waggle.data.vision import Camera

import numpy as np
from PIL import Image

# PyTorch imports 
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')

def capture_stream(plugin, stream, fps, nframes, out_dir=""):
    os.makedirs(out_dir, exist_ok=True)
    
    # use case 2
    with Camera() as camera:
        i=0
        for sample in camera.stream():
            # Save image
            sample_path = os.path.join(out_dir, datetime.now().astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S%z')+f"-{i:02d}.jpg")
            sample.save(sample_path.replace(':',''))

            time.sleep(1.0/fps)
            i = i+1
            
            if i > nframes: 
                break
	    
    print(f"Captured {nframes} images")
   
   
def capture(plugin, stream, fps, nframes, out_dir=""):
    os.makedirs(out_dir, exist_ok=True)
   
    # use case 1
    for i in range(nframes):
	    # Capture image
	    try:
	        sample = Camera().snapshot()
	    except:
	        print(f"Error capturing image. Simulating.")
	        sample = np.random.rand(100,100,3) * 255
	        sample = Image.fromarray(sample.astype('uint8'))
	    
	    # Save image
	    sample_path = os.path.join(out_dir, datetime.now().astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S%z')+f"-{i:02d}.jpg")
	    sample.save(sample_path)
    
	    time.sleep(1.0/fps)
	    
    print(f"Captured {nframes} images")
     
            
        
def main(args):

    FRAMES_FOLDER = os.path.normpath("FRAMES") # without last '/'
    CLASSIF_FOLDER = os.path.normpath("OUTPUT")
    NFRAMES = args.n
    
    os.makedirs(FRAMES_FOLDER, exist_ok=True)
    os.makedirs(CLASSIF_FOLDER, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Load CNN
    # ------------------------------------------------------------------
    print(f'[INFO] Loading model')
    # Setting the device to use for computations ('cuda' indicates GPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Valid versions are MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e or MDV6-rtdetr-c
    # TO DO: add model file to Docker
    detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov10-e")
   
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    
    # 1- Capture frames
    with Plugin() as plugin:
            capture_stream(plugin, args.stream, 25, NFRAMES, FRAMES_FOLDER)
            #capture(plugin, args.stream, 25, NFRAMES, FRAMES_FOLDER)
                     
    # 2- CNN classification
    results = detection_model.batch_image_detection(FRAMES_FOLDER, batch_size=16)
    json_file = os.path.join(CLASSIF_FOLDER, "detection_results.json")
    pw_utils.save_detection_json(results, json_file,
                             categories=detection_model.CLASS_NAMES,
                             exclude_category_ids=[], # Category IDs can be found in the definition of each model.
                             exclude_file_path=FRAMES_FOLDER)

    # a) Separate the positive and negative detections through file copying:
    if not args.annotations:
        pw_utils.detection_folder_separation(json_file, FRAMES_FOLDER, CLASSIF_FOLDER, args.threshold)
    # b) Save detections as annotated images
    else:
        pw_utils.save_detection_images(results, CLASSIF_FOLDER, overwrite=False)
    
    # 3- Publish detections 
    meta = {"camera":  f"{args.stream}"}
    with Plugin() as plugin:
        for r in results:
            if r['labels']: # if there is at least one detection
                if not any('animal' in e for e in r['labels']):
                    continue  # no animal detection ('human' or 'vehicle')
                plugin.publish(f'env.megadetector.animal', float(r['detections'].confidence[0]), 
                                  timestamp=int(os.path.getmtime(r['img_id'])*1e9), meta=meta)
                plugin.upload_file(r['img_id'], timestamp=int(os.path.getmtime(r['img_id'])*1e9), meta=meta)
     
    # Remove images and json 
    shutil.rmtree(FRAMES_FOLDER)
    shutil.rmtree(CLASSIF_FOLDER)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n', 
        action='store', default=5, type=int,
        help='Number of frames to capture')
    parser.add_argument(
        '--stream', dest='stream',
        help='ID or name of a stream', default='node-cam')
    parser.add_argument('--threshold', 
        type=float, default='0.2', 
        help='Confidence threshold to consider a detection as positive')
    parser.add_argument('--annotations', action='store_true', 
        help='Saving images annotated instead of raw images')
           
    args = parser.parse_args()
    exit(main(args))






