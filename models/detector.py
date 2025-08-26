import cv2
import torch
import torchvision.transforms as T
from models.classifier import OVDBoxClassifier, OVDMaskClassifier
from utils_dir.rpn_utils import get_box_RPN
from utils_dir.processing_utils import filter_boxes
from utils_dir.nms import non_max_suppression
from utils_dir.backbones_utils import prepare_image_for_backbone
from models.rpn.obb_rpn import OBBRPN
from models.rpn.box_rpn import BoxRPN
from utils_dir.area import topk_anchor_area
from utils_dir.write2file import write2file, write_cosine2file, write_cosine2file1
from utils_dir.visualizer_anchor import single_anchor_visualization

class OVDDetector(torch.nn.Module):
    def __init__(self,
                prototypes,
                bg_prototypes=None,
                backbone_type='dinov2',
                target_size=(602,602),
                scale_factor=2,
                min_box_size=5,
                ignore_index=-1,
                rpn_config='configs/FasterRCNN_FPN_DOTA_config.yaml',
                rpn_checkpoint='/data_student_2/zhouzhi/FSL_object_detection/ovdsat/data/weights/FasterRCNN_FPN_DOTA_final_model.pth',
                # rpn_checkpoint='data/weights/FasterRCNN_FPN_DOTA_final_model.pth',
                classification='box'    # Possible values: 'box', 'obb', 'mask'
                ):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.min_box_size = min_box_size
        self.ignore_index = ignore_index
        self.class_names = prototypes['label_names']
        self.num_classes = len(self.class_names)
        self.backbone_type = backbone_type        

        if classification not in ['box', 'mask', 'obb']:
            raise ValueError('Invalid classification type. Must be either "box", "obb" or "mask"')
        self.classification = classification

        if bg_prototypes is not None:
            all_prototypes = torch.cat([prototypes['prototypes'], bg_prototypes['prototypes']]).float()
        else:
            all_prototypes = prototypes['prototypes']

        # Initialize RPN
        if self.classification == 'box':
            self.rpn = BoxRPN(rpn_config, rpn_checkpoint)
        elif self.classification == 'obb':
            self.rpn = OBBRPN(rpn_config, rpn_checkpoint)
        elif self.classification == 'mask':
            raise NotImplementedError('Mask RPN not implemented yet. Should use SAM to generate proposals.')

        # Initialize Classifier
        classifier = OVDBoxClassifier if classification == 'box' else OVDMaskClassifier
        self.classifier = classifier(all_prototypes, prototypes['label_names'], backbone_type, target_size, scale_factor, min_box_size, ignore_index)

    def filter_and_nms(self, image_path, proposals, preds, proposals_scores, target_size, num_classes, iou_thr, conf_thres, box_conf_threshold, with_cosine, with_nms=True):
        B, _= proposals_scores.shape

        processed_predictions = []
        for b in range(B):
            img_file = image_path[b]
            # Filter and prepare boxes for NMS
            filtered_boxes, filtered_classes, filtered_scores = filter_boxes(
                                                                            img_file,
                                                                            proposals[b],
                                                                            preds[b],
                                                                            proposals_scores[b],
                                                                            target_size,
                                                                            num_classes,
                                                                            box_conf_threshold,
                                                                            with_cosine)
            
            # Use the cosine similarity class score as box confidence scores
            max_cls_scores, _ = torch.max(filtered_classes, dim=-1)
            
            # if with_cosine:
            #    write_cosine2file1('run/correct_and_preds_simd/after_with_cosine3.txt', max_cls_scores)
            #    write_cosine2file1('run/correct_and_preds_simd/after_with_cosine3_attnsum.txt', filtered_scores)
            #     filtered_scores = max_cls_scores

            pred_boxes_with_scores = torch.cat([filtered_boxes, filtered_scores[:, None], filtered_classes], dim=1)
            # print(f'pred_boxes_with_scores.shape is {pred_boxes_with_scores.shape}')
            
            sorted_indices = torch.argsort(filtered_scores, descending=True)
            pred_boxes_with_scores = pred_boxes_with_scores[sorted_indices]

            # Apply non maximum suppression
            nms_results = non_max_suppression(pred_boxes_with_scores.unsqueeze(0), iou_thres=iou_thr, conf_thres=conf_thres, with_nms=with_nms)

            # if with_cosine:
            #     single_anchor_visualization(img_file, nms_results[0], "after_second_nms_attn")
            processed_predictions.append(nms_results[0])

        return processed_predictions


    def forward(self, images, image_path, iou_thr=0.2, conf_thres=0.001, box_conf_threshold=0.01, aggregation='mean', labels=None):
        '''
        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
            iou_thr (float): IoU threshold for NMS
            conf_thres (float): Confidence threshold for NMS
            box_conf_threshold (float): Confidence threshold for box proposals
        '''
        with torch.no_grad():
            # Generate box proposals
            if self.classification == 'box':
                # _, proposals, proposals_scores = self.rpn(images)

                """-----------------------------"""
                logits, attention = self.rpn(images)
                logits_boxes, logits_box_scores = logits
                attention_boxes, attention_box_scores = attention
                # area = topk_anchor_area(attention_boxes).to(attention_box_scores.device)
                # attention_box_scores = attention_box_scores / area
                """-----------------------------"""

            elif self.classification == 'obb':
                boxes, proposals_scores, proposals = self.rpn(images)
            elif self.classification == 'mask':
                raise NotImplementedError('Mask RPN not implemented yet. Should use SAM to generate proposals.')  
            
            # Classify boxes with classifier
            # B, num_proposals = proposals_scores.shape
            """-----------------------------"""
            assert logits_box_scores.shape == attention_box_scores.shape
            B, num_proposals = logits_box_scores.shape
            """-----------------------------"""
            
            # 将proposals送入分类器进行分类，分类器返回的是一个批次图片的box聚合的相似度
            # preds = self.classifier(prepare_image_for_backbone(images, self.backbone_type), proposals, normalize=True, aggregation=aggregation)

            """-----------------------------"""
            logits_preds, attention_preds = self.classifier(prepare_image_for_backbone(images, self.backbone_type), logits_boxes, attention_boxes, normalize=True, aggregation=aggregation)
            """-----------------------------"""

            if num_proposals == 0:
                return [torch.tensor([], device=images.device) for _ in range(B)]
            
            # Extract class scores and predicted classes
            """-----------------------------"""
            logits_preds = logits_preds.reshape(B, num_proposals, -1)    
            attention_preds = attention_preds.reshape(B, num_proposals, -1)

            logits_classes = torch.argmax(logits_preds, dim=-1)
            attention_classes = torch.argmax(attention_preds, dim=-1)

            """-----------------------------"""
   
            # """
            #    处理预测结果，将其转换为合适的形状
            #    计算每个预测的得分，并根据得分获取相应的类别索引                                      
            # """
            # preds = preds.reshape(B, num_proposals, -1)    
            # scores, _ = torch.max(preds, dim=-1)    # 通过torch.max计算preds张量中每个提议的最大值
            # """
            # 举个例子：
            # preds = torch.randn(4, 100, 80)  # 假设 B=4, num_proposals=100, num_classes=80
            # scores, _ = torch.max(preds, dim=-1)  # 计算每个提议的最大得分
            # print(scores.shape)  # 输出: torch.Size([4, 100])
            # """
            # classes = torch.argmax(preds, dim=-1)   # 计算每个提议的类别索引，即选择最大得分所在的类别
            # """
            # 举个例子：
            # preds = torch.randn(4, 100, 80)  # 假设 B=4, num_proposals=100, num_classes=80
            # classes = torch.argmax(preds, dim=-1)  # 获取最大得分对应的类别索引
            # print(classes.shape)  # 输出: torch.Size([4, 100])
            # """
            # Filter predictions and prepare for NMS

            """-------------------------------------"""
            logits_pred = self.filter_and_nms(image_path, logits_boxes, logits_preds, logits_box_scores,
                                                     self.target_size, self.num_classes, iou_thr, conf_thres, box_conf_threshold, with_cosine=False, with_nms=True)
            attention_pred = self.filter_and_nms(image_path, attention_boxes, attention_preds, attention_box_scores, 
                                                       self.target_size, self.num_classes, iou_thr, conf_thres, box_conf_threshold, with_cosine=True, with_nms=True)
            """-------------------------------------"""
            del logits_preds, logits_classes, logits_boxes, logits_box_scores
            del attention_preds, attention_classes, attention_boxes, attention_box_scores

            # if 0:
            #     processed_predictions = []
            #     for b in range(B):
            #         # Filter and prepare boxes for NMS
            #         filtered_boxes, filtered_classes, filtered_scores = filter_boxes(proposals[b] if self.classification == 'box' else boxes[b],
            #                                                                         preds[b],
            #                                                                         proposals_scores[b],
            #                                                                         self.target_size,
            #                                                                         self.num_classes,
            #                                                                  box_conf_threshold)
            
            #         # Use the cosine similarity class score as box confidence scores
            #         max_cls_scores, _ = torch.max(filtered_classes, dim=-1)
            #         filtered_scores = max_cls_scores
            #         pred_boxes_with_scores = torch.cat([filtered_boxes, filtered_scores[:, None], filtered_classes], dim=1)

            #         # 返回按指定维度排序的张量索引
            #         # filtered_scores 是每个提议的得分（例如预测框的置信度分数）
            #         sorted_indices = torch.argsort(filtered_scores, descending=True)
            #         pred_boxes_with_scores = pred_boxes_with_scores[sorted_indices]
            #         # 包含预测坐标和得分的张量，sorted_indices是得分排序的索引，这一行代码通过这些索引重新排列pred_boxes_with_scores ，确保得分高的预测框排在前面。

            #         # Apply non maximum suppression
            #         nms_results = non_max_suppression(pred_boxes_with_scores.unsqueeze(0), iou_thres=iou_thr, conf_thres=conf_thres)
                    
            #         processed_predictions.append(nms_results[0])

            #     del preds, scores, classes, proposals, proposals_scores
            #     if self.classification == 'obb':
            #         del boxes
            #     return processed_predictions
            
            return logits_pred, attention_pred
        
            