import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import init_dataloaders
from models.detector import OVDDetector
from datasets import get_base_new_classes
from utils_dir.nms import custom_xywh2xyxy
from utils_dir.metrics import ap_per_class, box_iou
from utils_dir.processing_utils import map_labels_to_prototypes

from proposal_save.visualizer_proposal import visualizer_proposal
from utils_dir.write2file import write2file, write_cosine2file, write_cosine3file
from utils_dir.visualizer_anchor import single_anchor_visualization

# torch.empty_cache()
def prepare_model(args):
    '''
    Loads the model to evaluate given the input arguments and returns it.
    
    Args:
        args (argparse.Namespace): Input arguments
    '''

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:2")
    else:
        device = torch.device("cpu")

    # Load prototypes and background prototypes
    prototypes = torch.load(args.prototypes_path)
    bg_prototypes = torch.load(args.bg_prototypes_path) if args.bg_prototypes_path is not None else None
    model = OVDDetector(prototypes, bg_prototypes, scale_factor=args.scale_factor, backbone_type=args.backbone_type, target_size=args.target_size, classification=args.classification).to(device)
    #model.eval() 
    return model, device


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def replace_column(arr, new_column):
    for i in range(len(arr)):
        arr[i][4] = new_column[i]
    return arr

def process_preds(image_path, logits_preds, attention_preds, labels, iouv, boxes, device):
    stat = []
    for si, (logit_pred, attention_pred) in enumerate(zip(logits_preds, attention_preds)):
        image_file = image_path[si]
        keep = labels[si] > -1                                   
        targets = labels[si, keep]                               
        nl, logit_npr, attention_npr = targets.shape[0], logit_pred.shape[0], attention_pred.shape[0]
        
        #----------------------- 构造初始correct ----------------------
        logit_correct = torch.zeros(logit_npr, len(iouv), dtype=torch.bool, device=device)  # init
        attention_correct = torch.zeros(attention_npr, len(iouv), dtype=torch.bool, device=device)

        logit_predn = logit_pred.clone()  
        attention_preden = attention_pred.clone()
        # write_cosine2file('run/correct_and_preds_simd/after_second_nms3.txt', attention_preden)
        # write_cosine3file('run/correct_and_preds_simd/after_second_nms3_attnsum.txt', attention_preden)

        if logit_npr == 0:        # 如果没有预测框
            if nl:                # 如果没有目标标签
                tbox = custom_xywh2xyxy(boxes[si, keep, :])  
                labelsn = torch.cat((targets[..., None], tbox), 1) 

                # #----------------------- 经过预测correct ---------------------- 
                attention_correct = process_batch(attention_preden, labelsn, iouv)

                # #----------------------- 过滤掉利用attention预测时，全为False的预测结果 ----------------------
                # keep = attention_correct.any(dim=1)
                # attention_correct = attention_correct[keep]
                # attention_pred = attention_pred[keep]
                # single_anchor_visualization(image_file, attention_pred, "best_result_attn")
                # write_cosine2file('run/correct_and_preds_simd/bad1.txt', attention_pred_txt)
                # write_cosine3file('run/correct_and_preds_simd/best3_attnsum.txt', attention_pred_txt)

                stat.append((attention_correct, attention_pred[:, 4], attention_pred[:, 5], targets[:]))
                continue    

        if nl:   
            tbox = custom_xywh2xyxy(boxes[si, keep, :])  
            labelsn = torch.cat((targets[..., None], tbox), 1)  
            
            #----------------------- 经过预测correct ----------------------
            logit_correct = process_batch(logit_predn, labelsn, iouv)  
            attention_correct = process_batch(attention_preden, labelsn, iouv)

            # #----------------------- 过滤掉利用attention预测时，全为False的预测结果 ----------------------
            # keep = attention_correct.any(dim=1)
            # attention_correct = attention_correct[keep]
            # attention_pred = attention_pred[keep]
            
            # single_anchor_visualization(image_file, attention_pred, "best_result_attn")
            # write_cosine2file('run/correct_and_preds_simd/bad1.txt', attention_pred_txt)
            # write_cosine3file('run/correct_and_preds_simd/best3_attnsum.txt', attention_pred_txt)
            
            #----------------------- 过滤掉attention预测和logit预测重合的结果 ----------------------
            logit_pred_box = logit_pred[:, 0:4].unsqueeze(0).expand(attention_pred[:, 0:4].size(0), -1, -1)
            attention_pred_box = attention_pred[:, 0:4].unsqueeze(1).expand(-1, logit_pred[:, 0:4].size(0), -1)
            keep = ~torch.all(logit_pred_box==attention_pred_box, dim=2).any(dim=1) 

            attention_correct = attention_correct[keep]
            attention_pred = attention_pred[keep]

            if attention_pred.shape[0] != 0:
                #-----------------------将过滤完的attention预测的结果和logit预测的结果结合起来----------------------
                logit_correct = torch.cat((logit_correct, attention_correct), dim=0)
                logit_pred = torch.cat((logit_pred, attention_pred), dim=0)
            
            #-----------------------Cos_NMS----------------------
            _, index = torch.sort(logit_pred[:, 6], descending=True)
            logit_correct = logit_correct[index]
            logit_pred = logit_pred[index]

            cosim = logit_pred[:, 6]
            logit_boxes = logit_pred[:, 0:4]
            i = torchvision.ops.nms(logit_boxes, cosim, 0.5)
            
            logit_correct = logit_correct[i]
            logit_pred = logit_pred[i]
        
        stat.append((logit_correct, logit_pred[:, 4], logit_pred[:, 5], targets[:]))

    return stat

def eval_detection(args, model, val_dataloader, device):
    seen = 0
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    nc = val_dataloader.dataset.get_category_number()
    names = model.classifier.get_categories()
    val_cate = val_dataloader.dataset.get_categories()

    stats = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):

            if args.classification != 'mask':
                images, boxes, labels, metadata = batch
                boxes = boxes.to(device)
            else:
                images, _, labels, masks, _ = batch
                loc = masks.float().to(device)
          
            print(metadata["impath"])

            B = images.shape[0]

            labels = map_labels_to_prototypes(val_dataloader.dataset.get_categories(), model.classifier.get_categories(), labels)
            images = images.float().to(device)
            labels = labels.to(device)

            # preds = model(images, iou_thr=args.iou_thr, conf_thres=args.conf_thres, aggregation=args.aggregation)
            """------------------------"""
            logits_preds, attention_preds = model(images, metadata["impath"], iou_thr=args.iou_thr, conf_thres=args.conf_thres, aggregation=args.aggregation)
            process_stats = process_preds(metadata["impath"], logits_preds, attention_preds, labels, iouv, boxes, device)
            # visualizer_proposal(metadata["impath"], process_stats)
            stats.extend(process_stats)
            seen += B
            """------------------------"""                
            # for si, pred in enumerate(preds):
            #     keep = labels[si] > -1                                   # 保留所有标签值大于-1的标签，用于过滤掉不需要的标签
            #     targets = labels[si, keep]                               # 获取保留的目标标签
            #     nl, npr = targets.shape[0], pred.shape[0]  # number of labels, predictions
            #     correct = torch.zeros(npr, len(iouv), dtype=torch.bool, device=device)  # init
            #     seen += 1

            #     if npr == 0:   # 如果没有预测框
            #         if nl:     # 如果没有目标标签
            #             stats.append((correct, *torch.zeros((2, 0), device=device), targets[:]))
            #         continue    # 继续下一个样本的处理
                    
            #     predn = pred.clone()   # 克隆预测框数据，避免在后续操作中修改原数据
            #     if nl:   # 如果有目标标签
            #         tbox = custom_xywh2xyxy(boxes[si, keep, :])  # target boxes   #将目标框的坐标从xywh格式转换为xyxy格式
            #         labelsn = torch.cat((targets[..., None], tbox), 1)  # native-space labels   # 将标签和转换后的框坐标合并成一个包含标签和框位置的数组
                    
            #         correct = process_batch(predn, labelsn, iouv)  # 调用process_batch函数来计算预测框和真实框的匹配情况，返回正确的预测信息

            #     stats.append((correct, pred[:, 4], pred[:, 5], targets[:]))  # 将每个样本的统计信息添加到stats 

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    print(s)
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        filename = 'results1_{}.txt'.format(args.backbone_type)
        save_file_path = os.path.join(args.save_dir, filename)
        base_classes, new_classes = get_base_new_classes(args.dataset)

        with open(save_file_path, 'w') as file:
            file.write('Class Images Instances P R mAP50 mAP50-95\n')
            file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % ('all', seen, nt.sum(), mp, mr, map50, map))

            if nc > 1 and len(stats):
                map50_base = map_base = mr_base = mp_base = 0
                map50_new = map_new = mr_new = mp_new = 0
                for i, c in enumerate(ap_class):
                    file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
                    if names[c] in base_classes:
                        map50_base += ap50[i]
                        map_base += ap[i]
                        mr_base += r[i]
                        mp_base += p[i]
                    elif names[c] in new_classes:
                        map50_new += ap50[i]
                        map_new += ap[i]
                        mr_new += r[i]
                        mp_new += p[i]
                map50_base /= len(base_classes)
                map_base /= len(base_classes)
                mr_base /= len(base_classes)
                mp_base /= len(base_classes)
                map50_new /= len(new_classes)
                map_new /= len(new_classes)
                mr_new /= len(new_classes)
                mp_new /= len(new_classes)
                file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % ('total base', seen, nt.sum(), mp_base, mr_base, map50_base, map_base))
                file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % ('total new', seen, nt.sum(), mp_new, mr_new, map50_new, map_new))

def main(args):
    print('Setting up evaluation...')

    # Initialize dataloader
    _, val_dataloader = init_dataloaders(args)

    # Load model
    model, device = prepare_model(args)

    # Perform training
    eval_detection(
        args, 
        model, 
        val_dataloader, 
        device
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--val_root_dir', type=str)
    parser.add_argument('--val_annotations_file', type=str)
    parser.add_argument('--annotations', type=str, default='box')
    parser.add_argument('--prototypes_path', type=str)
    parser.add_argument('--bg_prototypes_path', type=str, default=None)
    parser.add_argument('--aggregation', type=str, default='mean')
    parser.add_argument('--classification', type=str, default='box')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    parser.add_argument('--iou_thr', type=float, default=0.2)
    parser.add_argument('--conf_thres', type=float, default=0.001)
    args = parser.parse_args()
    main(args)