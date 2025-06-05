import os

from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Load a model
model = YOLO("yolov8m.pt")  # build a new model from scratch

# Use the model
if __name__ == '__main__':
    results = model.train(data=os.path.join(ROOT_DIR, "mydataset.yaml"), pretrained=False, epochs=1000, imgsz=640, workers=0, batch=8)  # train the model
