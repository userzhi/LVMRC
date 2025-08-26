import torch
import numpy as np


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    box1 = box1[0: 4]
    box2 = box2[0: 4]
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def calculate_iou(box1, box2):
    # 计算两个框的交集部分
    box1 = box1[0: 4]
    box2 = box2[0: 4]
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # 计算交集区域的边界
    x_min_intersection = max(x_min1, x_min2)
    y_min_intersection = max(y_min1, y_min2)
    x_max_intersection = min(x_max1, x_max2)
    y_max_intersection = min(y_max1, y_max2)

    # 判断是否有交集
    intersection_width = max(0, x_max_intersection - x_min_intersection)
    intersection_height = max(0, y_max_intersection - y_min_intersection)

    intersection_area = intersection_width * intersection_height

    # 计算两个框的面积
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area
    return iou

def weighted_merge(g, p):
    g[:4] = (g[-1] * g[:4] + p[-1] * p[:4])/(g[-1] + p[-1])
    g[-1] = (g[-1] + p[-1]) / 2
    return g 

def anchor_area(bboxes):
    """
       bboxes是一个二维张量。
    """
    box_area = []
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        area.cpu().numpy()
        box_area.append(area)
    
    return box_area


def locality_aware_nms(boxes, classes, scores, cosim, iou_thread):
    
    # scores, index = torch.sort(scores)
    # boxes = boxes[index]
    s = []
    p = None
    prediction = torch.cat([boxes, classes, scores[:, None], cosim[:, None]], dim=1)
    for i, g in enumerate(prediction):

        if p is not None and calculate_iou(p, g) > iou_thread:
           p = weighted_merge(g, p)
        else:
            if p is not None:
               s.append(p)
            p = g 

        if p is not None:
            s.append(p)
    
    if len(s) == 0:
        return torch.empty(0)

    s = torch.stack(s)
    return s
