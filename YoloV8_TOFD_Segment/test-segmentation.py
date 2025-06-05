from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
import cv2
import numpy as np
import os

def predict_on_image(model, img, conf):
    result = model(img, conf=conf)[0]
    # result.boxes.xyxy   # box with xyxy format, (N, 4)
    cls = result.boxes.cls.cpu().numpy()   # cls, (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)

    # segmentation
    masks = None
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # masks, (N, H, W)
        masks = np.moveaxis(masks, 0, -1)  # masks, (H, W, N)
        # rescale masks to original image
        masks = scale_image(masks, result.masks.orig_shape)
        masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

    return boxes, masks, cls, probs


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

# Load a model
model = YOLO('best-segmentation.pt')

for root, dirs, files in os.walk("E:/TOFD-recognition/Images/", topdown=False):
    for name in files:
        filename = os.path.join(root, name)
        print(filename)
        img = cv2.imread(filename)
        # predict by YOLOv8
        boxes, masks, cls, probs = predict_on_image(model, img, conf=0.25)
        # overlay masks on original image
        image_with_masks = np.copy(img)
        if masks is not None:
            i = 0
            for mask_i in masks:
                color_mask = (255, 0, 0)
                if cls[i] == 1:
                    color_mask = (0, 255, 0)
                image_with_masks = overlay(image_with_masks, mask_i, color=color_mask, alpha=0.3)
                i = i + 1

        # Saving the image
        # cv2.imwrite('E:/Work/Project/Images/01_13C_TOFD_A-Scan_01_TOFD_A-Scan_with_mask.png', image_with_masks)
        cv2.imshow("output", image_with_masks)
        cv2.waitKey(0)





