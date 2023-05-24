# animal and bird detetction

import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import threading
import pygame


MOTORCYCLE_SOUND_FILE="D:\hackathon\Bird detection\motorcycle_sound.wav"
pygame.mixer.init()
motorcycle_sound = pygame.mixer.Sound(MOTORCYCLE_SOUND_FILE)

ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )
    sound_playing = False
    
    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[(detections.class_id == 14) | (detections.class_id == 17) | (detections.class_id == 18) | (detections.class_id == 19) | (detections.class_id == 20) | (detections.class_id == 21) | (detections.class_id == 22) | (detections.class_id == 23)]

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        if len(detections) > 0:
          if not sound_playing:
            motorcycle_sound.play(-1)  # Play sound in a loop
            sound_playing = True 
        else:
          if sound_playing:
            motorcycle_sound.stop()
            sound_playing = False    
      
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()
    
    