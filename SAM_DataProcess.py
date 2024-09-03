import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm
import glob

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamAutomaticMaskGenerator
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

seg_colors = np.zeros((32,3))
for c in range(32):
    seg_colors[c,0] = 255 - ((c+1) % 3)*85
    seg_colors[c,1] = 255 - (((c+1) // 3) % 3)*85
    seg_colors[c,2] = 255 - (((c+1) // 9) % 3)*85


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list, id):#tags_chinese
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask_%d.jpg'%(id)), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        # 'tags_chinese': tags_chinese,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)
    

def find_bound_box(mask):

    region_x, region_y = np.where(mask==True)[0], np.where(mask==True)[1]
    min_x = region_x[np.argmin(region_x)]
    max_x = region_x[np.argmax(region_x)]
    min_y = region_y[np.argmin(region_y)]
    max_y = region_y[np.argmax(region_y)]

    return min_x, max_x, min_y, max_y


def parse_mask_region(img, output_dir, masks_all, id):

    mask_img_all = img.copy()

    for idx, mask in enumerate(masks_all):

        mask_np = np.array(mask['segmentation'])

        # init general canvas
        mask_img = np.zeros(mask_np.shape)
        mask_img[mask_np == True] = 255
        mask_img_all[mask_np == True] = seg_colors[idx]
        img_filtered = img.copy()
        img_filtered[mask_np == False, :] = 0
        # save mask region
        cv2.imwrite(os.path.join(output_dir, 'general_mask','%d/%d.jpg'%(idx,id)), mask_img)
        # save mask img
        cv2.imwrite(os.path.join(output_dir, 'general_img','%d/%d.jpg'%(idx,id)), img_filtered)

        # init local canvas
        min_x, max_x, min_y, max_y = find_bound_box(mask_np)
        # min_x, min_y, max_x, max_y = mask['bbox']
        mask_img_cropped = mask_img[min_x:max_x,min_y:max_y]
        img_filtered_cropped = img_filtered[min_x:max_x,min_y:max_y]
        # save mask region
        cv2.imwrite(os.path.join(output_dir, 'local_mask','%d/%d.jpg'%(idx,id)), mask_img_cropped)
        # save mask img
        cv2.imwrite(os.path.join(output_dir, 'local_img','%d/%d.jpg'%(idx,id)), img_filtered_cropped)

    cv2.imwrite(os.path.join(output_dir,'%d.jpg'%(id)), mask_img_all)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("SAM_DataProcess", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.1, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.1, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    # image_path = args.input_image
    split = args.split
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        predictor = SamAutomaticMaskGenerator(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamAutomaticMaskGenerator(model=build_sam(checkpoint=sam_checkpoint).to(device),
                                              points_per_side=16, # 32
                                              min_mask_region_area=100, # None
                                              pred_iou_thresh=0.95 # 0.88
                                              ) 

    # build loop
    image_paths = glob.glob('image_dataset' + '/*.jpg')

    # make folders
    for f in range(32):
        os.makedirs('%s/general_mask/%d'%(output_dir,f),exist_ok=True)
        os.makedirs('%s/local_mask/%d'%(output_dir,f),exist_ok=True)
        os.makedirs('%s/general_img/%d'%(output_dir,f),exist_ok=True)
        os.makedirs('%s/local_img/%d'%(output_dir,f),exist_ok=True)
    os.makedirs('%s/ram'%(output_dir),exist_ok=True)

    for idx,image_path in enumerate(image_paths):

        image = cv2.imread(image_path)

        # output: 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'
        masks_all = predictor.generate(image)

        parse_mask_region(image, output_dir, masks_all, idx)


# python SAM_DataProcess.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --sam_checkpoint sam_vit_h_4b8939.pth --output_dir "outputs" --device "cuda"