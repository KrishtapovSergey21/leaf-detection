import cv2
from yolov8 import YOLOv8
import os

# Initialize yolov8 object detector
model_path = "best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Read image


for root, dirs, files in os.walk("../dataset/images/train/", topdown=False):
    for name in files:   
        filename = os.path.join(root, name)
        print(filename)
        img = cv2.imread(filename)

        # Detect Objects
        class_list = ['lateral','backwall']
        boxes, scores, class_ids = yolov8_detector(img)
        for index, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = round(box[0]), round(box[1]), round(box[2]), round(box[3])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            img = cv2.putText(img, class_list[class_ids[index]], (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Detected Objects", img)

        cv2.waitKey(0)
