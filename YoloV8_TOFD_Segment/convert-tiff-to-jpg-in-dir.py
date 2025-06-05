import json
import os
import cv2


def labelme_seg_to_yolo(json_path, output_folder, class_mapping=None):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    image_path = os.path.join(os.path.dirname(json_path), data['imagePath'])
    image_width, image_height = data['imageWidth'], data['imageHeight']

    yolo_lines = []

    for shape in data['shapes']:
        label = shape['label']

        if class_mapping and label in class_mapping:
            label = class_mapping[label]

        # Extracting segmentation data
        yolo_line = f"{label}"
        for point in shape['points']:
            # Normalizing coordinates
            x = point[0] / image_width
            y = point[1] / image_height
            yolo_line = yolo_line + f" {x} {y}"
        yolo_line = yolo_line + f"\n"
        yolo_lines.append(yolo_line)

    # Writing to YOLO format text file
    output_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(json_path))[0] + '.txt')
    with open(output_filename, 'w') as yolo_file:
        yolo_file.writelines(yolo_lines)


if __name__ == "__main__":
    tiff_folder = "E:/TOFD-recognition/dataset-detection-class6"
    output_folder = "E:/TOFD-recognition/dataset-detection-class6/png-files"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(tiff_folder):
        if filename.endswith(".tiff"):
            tiff_path = os.path.join(tiff_folder, filename)
            # Writing to YOLO format text file
            output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(tiff_path))[0] + '.png')
            img = cv2.imread(tiff_path);
            cv2.imwrite(output_path, img)
