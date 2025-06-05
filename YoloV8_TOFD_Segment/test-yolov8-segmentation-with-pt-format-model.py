import cv2
from ultralytics import YOLO
import os
from PIL import Image
class_list = ["lateral", "backwall"]
model = YOLO("best-segmentation.pt")

for root, dirs, files in os.walk("E:/TOFD-recognition/Images", topdown=False):
    for name in files:
        filename = os.path.join(root, name)
        print(filename)
        img = cv2.imread(filename)
        results = model.predict(source=img)

        color = (255, 0, 0)    # Blue color in BGR
        thickness = 2  # Line thickness of 2 px
        isClosed = True
        for result in results:
            masks = result.masks  # get boxes on cpu in numpy
            if masks is not None:
                for mask in masks:  # iterate mask
                    segmentation = mask.data[0].numpy()
                    mask_img = Image.fromarray(segmentation, "I")
                    mask_img
                    #polygon = mask.xy[0].astype(int)  # get mask polygon points as int
                    #pts = polygon.reshape((-1, 1, 2))
                    #img = cv2.polylines(img, pts, isClosed, color, thickness)
        cv2.imshow("output", img)
        cv2.waitKey(0)
        print(name)