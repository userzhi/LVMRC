import numpy as np
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_and_save_proposals(img_file, predict_stat, cate):
    save_path='/data_student_2/zhouzhi/FSL_object_detection/ovdsat/result_visualization'
    
    for j, image_dir in enumerate(img_file):
        orignal_img = cv2.imread(image_dir)

        image = cv2.resize(orignal_img, (602, 602))
    # Plot proposals with top K objectness scores
        
        proposal_boxes = predict_stat[j][1].tolist()
    # proposal_boxes = [
    #     [3.0407e+02, 3.4356e+02, 4.0854e+02, 4.7822e+02, 4.8058e-01, 5.0000e+00,
    #      5.8440e-01],
    #     [1.9942e+02, 2.0581e+02, 3.0890e+02, 3.4355e+02, 4.1694e-01, 5.0000e+00,
    #      5.6370e-01],
    #     [7.1451e+01, 3.0852e+01, 1.8992e+02, 1.8411e+02, 1.5818e+02, 5.0000e+00,
    #      5.6272e-01]]
    # top_k = len(proposal_boxes)
    # # top_k = 20

        for i in range(len(proposal_boxes)):
            # print(proposal_boxes[i])
            if isinstance(proposal_boxes[i], (tuple, list)) and len(proposal_boxes[i]) >= 7:
                x, y, w, h, _, q, _= proposal_boxes[i]
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                cv2.rectangle(image, (x, y), (w, h), color=(0, 97, 255), thickness=2)
                cv2.putText(image, str(cate[int(q)]), (x-10, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 97, 255))
                # rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
                # ax.add_patch(rect)
        
        cv2.imwrite(os.path.join(save_path, image_dir.split('/')[-1].split('.')[0]+'result_'+'.jpg'), image)


def plot_classification_confusion_matrix(matrix, nc, normalize=True, save_dir='', names=()):
        import seaborn as sn
        import warnings
        import matplotlib.pyplot as plt
        from pathlib import Path

        array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
            
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)