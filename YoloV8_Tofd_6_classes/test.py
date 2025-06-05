from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(source="../dataset/images/train/20221214_000001.jpg")
print(results)
