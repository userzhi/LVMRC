import os
import cv2
import numpy as np

def batch_anchor_visualization(img_files, anchor_data, stage):
    """
       参1：list, 其中每一个元素代表图片路径
       参2：anchor数据，shape为[4, 1000, 4]
       参3：保存什么阶段时间的anchor
    """
    save_path = 'visualizer_anchor/simd'
    for i, img_file in enumerate(img_files):
        image = cv2.imread(img_file)
        if image is None:
            print("图片读取失败！")
        image = cv2.resize(image, (602, 602))
        
        image_name = img_file.split('/')[-1]

        anchors = anchor_data[i]
        for k in range(len(anchors)):
            assert len(anchors[k]) == 4, "检查anchors数据"
            x, y, w, h = [int(loc) for loc in anchors[k]]
            cv2.rectangle(image, (x, y), (w, h), color=(0, 97, 255), thickness=2)
            cv2.imwrite(os.path.join(save_path, stage, "attn_anchor_"+image_name), image)



def single_anchor_visualization(img_file, anchor_data, stage):
    """
       参1：string, 图片路径
       参2：anchor数据，shape为[N, 4], N表示剩余anchor个数
       参3：保存什么阶段时间的anchor
    """
    save_path = 'visualizer_anchor/simd'
    image = cv2.imread(img_file)
    if image is None:
         print("图片读取失败！")
    image = cv2.resize(image, (602, 602))
        
    image_name = img_file.split('/')[-1]
    for k in range(len(anchor_data)):
        if len(anchor_data[k]) == 4:
            x, y, w, h = [int(loc) for loc in anchor_data[k]]
        elif len(anchor_data[k]) == 7:
            x, y, w, h = [int(loc) for loc in anchor_data[k][0: 4]]
            
        cv2.rectangle(image, (x, y), (w, h), color=(0, 97, 255), thickness=2)
        cv2.imwrite(os.path.join(save_path, stage, "attn_anchor_"+image_name), image)





