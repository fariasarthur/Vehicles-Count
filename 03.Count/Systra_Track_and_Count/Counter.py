#Segunda versão do código de que realiza contagem de veículos
import torch
import cv2
import numpy as np
from collections import OrderedDict
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
#from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.dataloaders import letterbox
from yolov5.models.yolo import detect

class_names = ['car', 'bus', 'truck']
classes = [2, 5, 7] # Índices das classes de veículos em COCO dataset

class VehicleTracker:
    def __init__(self):
        self.tracker = cv2.TrackerKCF_create()

    def init_tracker(self, image, bbox):
        # Inicializa o tracker com o primeiro bbox
        self.tracker.init(image, bbox)

    def update_tracker(self, image):
        # Atualiza o tracker com a imagem atual
        success, bbox = self.tracker.update(image)
        if success:
            return bbox
        else:
            return None

def detect_and_count_vehicles(image, model, device, conf_threshold=0.5, iou_threshold=0.45):
    # Configuração das classes a serem detectadas
    #classes = [2, 5, 7] # Índices das classes de veículos em COCO dataset
    #class_names = ['car', 'bus', 'truck']

    # Resize da imagem e conversão para tensor
    img_size = model.img_size
    img = letterbox(image, img_size, stride=model.stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img.unsqueeze_(0)

    # Detecção de objetos utilizando o modelo YOLOv5
    detections = detect(img, model, conf_threshold, iou_threshold)[0]

    # Seleção dos objetos detectados que correspondem a veículos
    detections = detections[detections[:, 5].isin(classes)]

    # Contagem dos veículos por classe
    vehicle_count = {}
    for class_idx in classes:
        class_name = class_names[classes.index(class_idx)]
        vehicle_count[class_name] = (detections[:, 5] == class_idx).sum()

    return vehicle_count, detections

if __name__ == '__main__':
    # Inicialização do modelo YOLOv5
    weights = 'yolov5s.pt'
    device = select_device('')
    model = attempt_load(weights, map_location=device)
    model.eval()

    # Inicialização do tracker
    tracker = VehicleTracker()

    # Leitura do vídeo
    cap = cv2.VideoCapture('video.mp4')

    # Loop de processamento do vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            break

    vehicle_count, detections = detect_and_count_vehicles(frame, model, device)

    # Desenho dos bboxes e das classes dos veículos detectados
    if len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, class_idx = det.tolist()
            class_name = class_names[classes.index(int(class_idx))]
            label = f'{class_name}: {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Rastreamento dos veículos detectados
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])
            if not tracker.tracker.getObjects():
                tracker.init_tracker(frame, bbox)
            else:
                tracked_bbox = tracker.update_tracker(frame)
                if tracked_bbox is not None:
                    x, y, w, h = [int(coord) for coord in tracked_bbox]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Exibição da contagem de veículos
    for class_name in vehicle_count:
        count = vehicle_count[class_name]
        cv2.putText(frame, f'{class_name}: {count}', (10, 30*(classes.index(class_names.index(class_name))+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Exibição do frame
        cv2.imshow('Vehicle Detection and Counting', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
