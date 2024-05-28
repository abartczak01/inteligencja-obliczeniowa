from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

output_folder = "./detections"
os.makedirs(output_folder, exist_ok=True)

input_folder = "./images"

for filename in os.listdir(input_folder):

    input_path = os.path.join(input_folder, filename)

    results = model(input_path)

    img_with_boxes = results[0].plot()

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img_with_boxes)

print("koniec")


# YOLO (You Only Look Once)
# YOLO przetwarza obraz w całości podczas jednej przejściowej operacji sieci neuronowej. W przeciwieństwie do wcześniejszych podejść, które dzieliły obraz na regiony i 
# analizowały każdy region osobno, YOLO analizuje cały obraz na raz, co przyspiesza proces detekcji.
# Uproszczony schemat działania (w yolo_graph.png): od razu ze zdjęcia przechodzi do bounding boxes (ramki) i klasyfikacji.
# YOLO jest bardzo szybki stąd jest odpowiedni do np. pojazdów autonomicznych czy monitorowania obrazu w czasie rzeczywistym.

