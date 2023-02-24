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

    #  converter o quadro para RGB e criar um tensor
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))[None, ...]
    h, w, c = rgb_frame.shape[2], rgb_frame.shape[3], rgb_frame.shape[1]
    if c == 1:
        rgb_frame = cv2.merge([rgb_frame] * 3)  # adicionar 2 canais extras para escala de cinza
    elif c != 3:
        raise ValueError(f"Unexpected number of channels: {c}")
    
    tensor_frame = torch.from_numpy(rgb_frame)
    input_tensor = tensor_frame

    #pre-processamento

    # Definir tamanho da imagem de entrada do modelo
    input_size = 640

    # Redimensionar o tensor de entrada para o tamanho correto
    resize_transform = transforms.Resize((input_size, input_size))
    input_tensor = resize_transform(input_tensor)
    input_tensor = input_tensor.float()

    # Normalizar os valores dos pixels da imagem para o intervalo [0, 1]
    normalize_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    input_tensor = normalize_transform(input_tensor)

    # Transpor as dimensões do tensor para que fique no formato correto para o modelo
    #input_tensor = input_tensor.permute(0, 2, 3, 1)
    input_tensor_shape = input_tensor.shape

    # inference with larger input size
    results = model(input_tensor, size=input_size)

    # inference with test time augmentation
    results_aug = model(input_tensor, augment=True)

    # Rastrear veículos detectados
    detections = results.pred[0]
    tracker.update(detections)

    # Rastrear veículos detectados com o aumento de tempo de teste
    detections_aug = results_aug.pred[0]
    tracker.update(detections_aug)

    # Desenhe a linha de contagem no quadro atual
    cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

    # Percorra os veículos rastreados
    for vehicle in tracker.objects:
        # Verifique se o veículo cruzou a linha de contagem
        if vehicle.crossed_line(line):
            # Atualize a contagem de veículos para a classe correspondente
            vehicle_count[vehicle.cls] += 1

    # Desenhe a contagem atual de veículos na tela
    text = f"Car: {vehicle_count['car']} Bus: {vehicle_count['bus']} Truck: {vehicle_count['truck']}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar o quadro atual
    cv2.imshow('frame', frame)

    # Verifique se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

