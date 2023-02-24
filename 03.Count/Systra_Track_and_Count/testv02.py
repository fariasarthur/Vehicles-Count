import cv2
import numpy as np
import torch
import yolov5
from bytetracker import BYTETracker
import torchvision.transforms as transforms


# Inicialize o detector de objetos YOLOv5 com a base pré-treinada MS COCO
model = yolov5.load('yolov5s.pt')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# Inicialize o rastreador ByteTracker
tracker = BYTETracker()

# Defina a linha de contagem (vertical)
line = ((300, 0), (300, 720))

# Inicialize a contagem de veículos
vehicle_count = {'car': 0, 'bus': 0, 'truck': 0}

# Inicialize o vídeo de entrada
cap = cv2.VideoCapture('test_systra.mp4')

# Loop através do vídeo de entrada
while cap.isOpened():
    # Leitura do frame atual
    ret, frame = cap.read()
    
    if not ret:
        break

    results = model(frame, size=1280, augment=True)

    