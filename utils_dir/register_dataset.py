import os
import cv2
import numpy as np
from detectron2.structures import BoxMode


def obb_to_bbox(obb_yolo):
    
    obb_list = []
    for i in range(len(obb_yolo)):
        if i % 2 == 0:
            obb_list.append([float(obb_yolo[i]), float((obb_yolo[i+1]))])
            continue
    obb_list = np.array(obb_list)

    xmin = np.min(obb_list[:, 0])
    ymin = np.min(obb_list[:, 1])
    xmax = np.max(obb_list[:, 0])
    ymax = np.max(obb_list[:, 1])

    return [xmin, ymin, xmax, ymax]


def get_dota_dicts(data_dir, fun, category_dict):

    dataset_dicts = []
    img_dir = os.path.join(data_dir, fun, 'images')
    anno_dir = os.path.join(data_dir, fun, 'annotations')

    for root, dirs, files in os.walk(img_dir):
        for idx, image_file in enumerate(files):
            record = {}

            filename = os.path.join(img_dir, image_file)
            height, width = cv2.imread(filename=filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            anno_file = os.path.join(anno_dir, os.path.splitext(image_file)[0]+'.txt')
            objs = []
            with open(anno_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if "imagesource" not in line.split()[0] and "gsd" not in line.split()[0]:
                        obj = {
                            # "bbox": [float(x1) for x1 in line.split()[0: 8]],
                            "bbox": obb_to_bbox(line.split()[0: 8]),
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": category_dict[line.split()[8]],
                            "iscrowd":0
                        }
                objs.append(obj)
                record["annotations"] = objs

            dataset_dicts.append(record)

    return dataset_dicts


    
                


