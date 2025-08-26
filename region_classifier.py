import os
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('/data_student_2/zhouzhi/FSL_object_detection/ovdsat/tensorboard')
import os.path as osp
import torch.nn.functional as F
from torchvision import transforms
from argparse import ArgumentParser
from utils_dir.backbones_utils import load_backbone, extract_backbone_features, get_backbone_params


class RegionClassifier(nn.Module):
    def __init__(self, num_classes) -> None:
        super(RegionClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x) 
        x = self.relu1(x)
        x = self.fc2(x) 
        x = self.relu2(x)  
        x = self.fc3(x)
        return x

def initialize_weights(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)

        elif isinstance(layer, nn.Conv2d):
            init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            init.zeros_(layer.bias)

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  


def preprocess(image, mask=None, backbone_type='dinov2', target_size=(602, 602), patch_size=14):
    '''
    Preprocess an image and its mask to fed the image to the backbone and mask the extracted patches.

    Args:
        image (PIL.Image): Input image
        mask (PIL.Image): Input mask
        backbone_type (str): Backbone type
        target_size (tuple): Target size of the image
        patch_size (int): Patch size of the backbone
    '''

    if 'clip' in backbone_type:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Transform images to tensors and normalize
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize the images to a size larger than the window size
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # Normalize the images
    ])

    if mask is not None:
        m_w, m_h = target_size
        mask = transforms.Resize((m_w//patch_size, m_h//patch_size), interpolation=Image.NEAREST)(mask)

    image = transform(image).unsqueeze(0)
    return image, mask


def load_data(args):

    img_file = []
    for cls in os.listdir(args.data_dir):
        cls_data_path = os.path.join(args.data_dir, cls)
        for root, dirs, files in os.walk(cls_data_path):
            for x in files:
                if '.mask' not in x:
                    img_path = os.path.join(root, x)
                    img_file.append(img_path)

    return img_file

def from_img_to_instance(args, image, category):

    with open(args.annotations_file) as f:
        json_data = json.load(f)
    
    img_name = image.split('/')[-1]
    img_class = image.split('/')[-2]

    imgs = json_data["images"]
    annos = json_data["annotations"]

    instance_box = []
    for img in imgs:
        if img["file_name"] == img_name:
            img_id = img["id"]
            for anno in annos:
                if args.statu == "train":
                   if anno["image_id"] == img_id and anno["category_id"] == category.index(img_class):
                      instance_box.append(anno["bbox"])
                elif args.statu == "test":
                    if anno["image_id"] == img_id :
                      instance_box.append(anno["bbox"])
    return instance_box


def train_one_epoch(args, model, region_classifier, category, patch_size, criterion, optimizer):
    
    img_file = load_data(args)
    img_num = len(img_file)
    idxs = list(np.arange(img_num))
    np.random.shuffle(idxs)

    total_loss = 0
    object_num = 0
    for img_id in idxs:

        image = Image.open(img_file[img_id])
        img_class = img_file[img_id].split('/')[-2]
        gt_labels = torch.tensor([category.index(img_class)]).to(args.device)
       
        w, h = image.size
        w_new, h_new = args.target_size
        scale_x = w_new / w
        scale_y = h_new / h
        instance_box = from_img_to_instance(args, img_file[img_id], category)

        image, _= preprocess(image, backbone_type=args.backbone_type, target_size=args.target_size, patch_size=patch_size)
        features = extract_backbone_features(image.to(args.device), model, args.backbone_type, scale_factor=args.scale_factor)
        _, K, D = features.shape
        p_w = p_h = int(K**0.5)
        features = features.reshape(p_h, p_w, -1).permute(2, 0, 1).unsqueeze(0)
        
        for box in instance_box:
            x1, y1, w1, h1= box
            x2 = x1 * scale_x
            y2 = y1 * scale_y
            w2 = w1 * scale_x
            h2 = h1 * scale_y
            feat = features[:, :, int(y2 // patch_size):int((y2 + h2) // patch_size), int(x2 // patch_size):int((x2 + w2) // patch_size)]
            feat = feat.mean(dim=[2, 3])

            outputs = region_classifier(feat)  

            optimizer.zero_grad()
            loss = criterion(outputs, gt_labels)
            loss.backward()
            optimizer.step()
            
            object_num  += 1
            total_loss += loss.item()

    avg_loss = total_loss / object_num 
    return avg_loss

def evaluation(args, model, classifier, patch_size):

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_epoch45.pth')
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['checkpoints'])

    with torch.no_grad():
        image = Image.open(args.test_file)

        w, h = image.size
        w_new, h_new = args.target_size
        scale_x = w_new / w
        scale_y = h_new / h
        instance_box = from_img_to_instance(args, args.test_file, category)

        image, _= preprocess(image, backbone_type=args.backbone_type, target_size=args.target_size, patch_size=patch_size)
        features = extract_backbone_features(image.to(args.device), model, args.backbone_type, scale_factor=args.scale_factor)
        _, K, D = features.shape
        p_w = p_h = int(K**0.5)
        features = features.reshape(p_h, p_w, -1).permute(2, 0, 1).unsqueeze(0)
        
        for box in instance_box:
            x1, y1, w1, h1= box
            x2 = x1 * scale_x
            y2 = y1 * scale_y
            w2 = w1 * scale_x
            h2 = h1 * scale_y
            feat = features[:, :, int(y2 // patch_size):int((y2 + h2) // patch_size), int(x2 // patch_size):int((x2 + w2) // patch_size)]
            feat = feat.mean(dim=[2, 3])

            outputs = classifier(feat)
            print(outputs)

        
def main(args, category):
    
    set_random_seed(42)
    model = load_backbone(args.backbone_type)
    model = model.to(args.device)
    # model.eval()
    patch_size, _ = get_backbone_params(args.backbone_type)  
    classifier = RegionClassifier(args.num_classes).to(args.device)

    if args.statu == 'train':
        
        for name, parameter in model.named_parameters():
            if name in ["blocks.23.mlp.fc2.weight", "blocks.23.mlp.fc2.bias", 
                       "blocks.23.ls2.gamma", "norm.weight", "norm.bias"]:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
        model.train()

        initialize_weights(classifier)
        classifier.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001)

        for epoch in tqdm(range(args.num_epochs), desc="Training Epochs", ncols=100):
            classifier.train()
            train_loss = train_one_epoch(args, model, classifier, category, patch_size, criterion, optimizer)
            tqdm.write(f"Epoch {epoch}, Loss: {train_loss:.4f}")
            if epoch % 5 == 0:
                chenkpoint = {
                    "checkpoints": classifier.state_dict(),
                    "epoch": epoch
                }
                # writer.add_scalar('Loss/train', train_loss, epoch)
                torch.save(chenkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch}.pth'))
        
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'finetune_dinov2.pth'))
        # writer.close()
    
    elif args.statu == 'test':
        for name, parameter in classifier.named_parameters():
            parameter.requires_grad = False
        classifier.eval()
        for name, parameter in model.named_parameters():
            parameter.requires_grad = False
        model.eval()

        # model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'finetune_dinov2.pth')))
        evaluation(args, model, classifier, patch_size)
        

if __name__ == '__main__':
    
    category = ["car", "truck", "van", "long-vehicle", "bus", "airliner", "propeller-aircraft", "trainer-aircraft",
                "charted-aircraft", "figther-aircraft", "others", "stair-truck", "pushback-truck", "helicopter", "boat"]

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/init_data/simd_N10')
    # -------------------------在train和test不同模式需要改变------------------------
    parser.add_argument('--annotations_file', type=str, default='/data_student_2/zhouzhi/FSL_object_detection/ovdsat/data/simd/train_coco_subset_N10.json')
    # -------------------------------------------------

    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--statu', type=str, default='test')
    parser.add_argument('--test_file', type=str, default='/data_student_2/zhouzhi/FSL_object_detection/ovdsat/data/simd/val/4922.jpg')

    parser.add_argument('--checkpoint_dir', type=str, default='/data_student_2/zhouzhi/FSL_object_detection/ovdsat/checkpoint')
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(602, 602))
    parser.add_argument('--window_size', type=int, default=224)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--num_b', type=int, default=10, help='Number of background samples to extract per image')
    args = parser.parse_args()

    main(args, category)

