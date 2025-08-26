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

# Requirements

Create a Conda environment and install the required packages as follows:

<div style="max-height: 400px; overflow: auto; border: 1px solid #ddd; padding: 10px;">

<pre>
<code>
conda create -n LVMRC python=3.9 -y
conda activate LVMRC
pip install torch==1.13.0+cu116 \
            torchvision==0.14.0+cu116 \
            torchaudio==0.13.0 \
            --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python albumentations transformers
</code>
</pre>

</div>

# Data preparation
We provide the same splits and labels we use in our article for the SIMD dataset (N = {5, 10, 30}). The data path should follow the structure below for each dataset, e.g. simd, dior or your own:
<div style="max-height: 400px; overflow: auto; border: 1px solid #ddd; padding: 10px;">

<pre>
<code>
data/
│
├── simd/
│   ├── train_coco_subset_N5.json
│   ├── train_coco_subset_N10.json
│   ├── train_coco_subset_N30.json
│   ├── train_coco_finetune_val.json
│   ├── val_coco.json
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── dior/
│   ├── train_coco_subset_N5.json
│   ├── train_coco_subset_N10.json
│   ├── train_coco_subset_N30.json
│   └── ...
...
</code>
</pre>

</div>
