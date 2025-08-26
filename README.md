# LVMRC： Large Visual Model-Based Region Classifier for Few-Shot Remote Sensing Object Detection
# Abstract:
  Recently, a large-scale visual model named DINOv2 has demonstrated exceptional capabilities in feature representation for downstream computer vision tasks. Researchers have applied it to the field of few-shot object detection and proposed a DINOv2-based few-shot object detection paradigm, achieving impressive detection performance. However, category prototypes are constructed without explicit semantic guidance, which limits the performance of this paradigm. To address this problem, this paper proposes a region classification method based on large-scale visual model (LVMRC), aiming to classify region proposals in few-shot object detection. First, we propose a self-supervised features-based logits region classifier (SPLRC). SPLRC could capture the deep correlation between region features and class semantics through a nonlinear transformation mechanism, making it effectively enhance the ability of detection model to classify region proposals. To integrate the advantages of similarity-based and logits-based criteria, we develop a cascade-residual region classifier (CRRC). CRRC not only effectively leverages feature alignment capability of similarity-based metric and semantic representation power of logits-based criterion, but introduce a residual fusion mechanism to establish a novel meta-criterion. CRRC further enhances the capability of detection model to classify region proposals. We carry out comprehensive experiments on DIOR and SIMD remote sensing datasets to demonstrate that the proposed method achieves competitive performance. It is noteworthy that LVMRC are employed without fine-tuning any parameter.
# Overview
<p align="center">
  <img width="700" height="700" alt="image" src="https://github.com/user-attachments/assets/14b97009-3dce-4c39-9677-8da2604571ae" />
</p>
The pipeline of detection process:

1. Prepare the data with N labelled examples per category
2. Create Metric Learning-Based Prototype Region Classifier
3. Training Self-Supervised Features-Based Logits Region Classifier
4. Detect objects via RPN and two Region Classifiers 
# Self-Supervised Features-Based Logits Region Classifier（SPLRC）
<p align="center">
  <img src="https://github.com/userzhi/LVMRC/blob/main/images/logits.png?raw=true" width="600" />
</p>


<div style="max-height: 400px; overflow: auto; border: 1px solid #ddd; padding: 10px;">

```bash
# 1. 创建 Conda 环境
conda create -n ovdsat python=3.9 -y

# 2. 激活环境
conda activate ovdsat

# 3. 安装指定版本的 PyTorch（带 CUDA 11.6 支持）
pip install torch==1.13.0+cu116 \
            torchvision==0.14.0+cu116 \
            torchaudio==0.13.0 \
            --extra-index-url https://download.pytorch.org/whl/cu116

# 4. 安装 Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 5. 安装其他依赖库
pip install opencv-python albumentations transformers

# 6. 验证 CUDA 是否可用（可选）
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
</div>
