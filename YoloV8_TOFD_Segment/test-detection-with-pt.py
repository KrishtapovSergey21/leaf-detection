import cv2
from ultralytics import YOLO
import os
class_list = ["fault"]
model = YOLO("best-detection.pt")

for root, dirs, files in os.walk("E:/TOFD-recognition/Images/", topdown=False):
    for name in files:
        filename = os.path.join(root, name)
        print(filename)
        img = cv2.imread(filename)
        results = model.predict(source=img)

        for result in results:
            boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy

            for box in boxes:  # iterate boxes
                r = box.xyxy[0].astype(int)  # get corner points as int
                cls = result.names[int(box.cls[0])]
                cls = int(cls.replace("class_", ""))
                confidence = box.conf.item()
                if confidence > 0.5:
                    cv2.putText(img, class_list[cls], (int(list(r[:2])[0]), int(list(r[:2])[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(img, r[:2], r[2:], (0, 0, 255), 1)  # draw boxes on img
        cv2.imshow("output", img)
        cv2.waitKey(0)
        print(name)



