import torch
from utils_dir.visualizer_anchor import single_anchor_visualization

from utils_dir.locality_aware_nms import anchor_area
from utils_dir.write2file import list2file
from utils_dir.locality_aware_nms import locality_aware_nms

def filter_boxes(img_file, boxes, classes, scores, target_size, num_labels, box_conf_threshold, with_cosine):

    target_height, target_width = target_size
    keep = ((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) &
            (boxes[:, 2] <= target_width) & (boxes[:, 3] <= target_height))

    filtered_boxes = boxes[keep]
    filtered_classes = classes[keep]
    filtered_scores = scores[keep]
    # Filter out boxes classified as background
    predictions = torch.argmax(filtered_classes, dim=-1)    
    filtered_boxes = filtered_boxes[predictions < num_labels, ...]
    filtered_classes = filtered_classes[predictions < num_labels, ...]
    filtered_scores = filtered_scores[predictions < num_labels]

    # Filter out boxes with low confidence
    keep = filtered_scores > box_conf_threshold
    filtered_boxes = filtered_boxes[keep, ...]
    filtered_classes = filtered_classes[keep, ...]
    filtered_scores = filtered_scores[keep]
    
    # 得到余弦相似度
    if with_cosine:
        # single_anchor_visualization(img_file, filtered_boxes, "remove_background")
        # keep = filtered_scores < 200
        # filtered_boxes = filtered_boxes[keep, ...]
        # filtered_classes = filtered_classes[keep, ...]
        # filtered_scores = filtered_scores[keep]

        cos_sim, _ = torch.max(filtered_classes, dim=-1)
        keep = cos_sim > 0.7
        filtered_boxes = filtered_boxes[keep, ...]
        filtered_classes = filtered_classes[keep, ...]
        filtered_scores = filtered_scores[keep]
        
        if len(filtered_boxes) != 0:
            num_classes = len(filtered_classes[0])
            cos_sim, _ = torch.max(filtered_classes, dim=-1)
            s = locality_aware_nms(filtered_boxes, filtered_classes, filtered_scores, cos_sim, 0.45)
            filtered_boxes = s[:, 0: 4]
            filtered_classes = s[:, 4: num_classes]
            filtered_scores = s[:, num_classes].flatten()
            single_anchor_visualization(img_file, filtered_boxes, "after_lanms")
        

        # area = anchor_area(filtered_boxes)
        # area = torch.tensor(area)
        # keep = (area > 1000)&(area < 30000)
        # filtered_boxes = filtered_boxes[keep, ...]
        # filtered_classes = filtered_classes[keep, ...]
        # filtered_scores = filtered_scores[keep]

        # single_anchor_visualization(img_file, filtered_boxes, "after_area")

    return filtered_boxes, filtered_classes[:, :num_labels], filtered_scores


def map_labels_to_prototypes(dataset_categories, model_prototypes, labels):
    mapped_labels = []
    # Create a reverse mapping from class names to indices for the dataset categories
    dataset_categories_reverse = {v: k for k, v in model_prototypes.items()}
    # Map dataset category indices to model prototype indices
    for batch_labels in labels:
        mapped_batch_labels = []
        for label in batch_labels:
            if label == -1:
                mapped_batch_labels.append(-1)
            elif label.item() in dataset_categories and dataset_categories[label.item()] in dataset_categories_reverse:
                class_name = dataset_categories[label.item()]
                if class_name in dataset_categories_reverse:
                    mapped_batch_labels.append(dataset_categories_reverse[class_name])
                else:
                    mapped_batch_labels.append(-1)
            else:
                mapped_batch_labels.append(-1)
        mapped_labels.append(mapped_batch_labels)
    
    return torch.tensor(mapped_labels)