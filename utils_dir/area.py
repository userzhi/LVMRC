import torch


def topk_anchor_area(attention_boxes):
    bs = attention_boxes.shape[0]
    area = []
    
    for i in range(bs):
        area.append(torch.Tensor([((x[2] - x[0]) * (x[3] - x[1])) for x in attention_boxes[i].tolist()]))
    
    area = torch.stack(area)
    return area 