import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image
import glob
import cv2

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def find_bound_box(mask):

    region_x, region_y = np.where(mask==True)
    min_x = region_x[np.argmin(region_x)]
    max_x = region_x[np.argmax(region_x)]
    min_y = region_y[np.argmin(region_y)]
    max_y = region_y[np.argmax(region_y)]

    return min_x, max_x, min_y, max_y


def parse_mask_region(img, output_dir, mask_list, id):
    # value = 0  # 0 for background
    # plt.figure(figsize=(10, 10))

    for idx, mask in enumerate(mask_list):

        # init general canvas
        mask_img = torch.zeros(mask_list.shape[-2:])
        mask_img[mask.cpu().numpy()[0] == True] = 1
        img_filtered = img.copy()
        img_filtered[mask.cpu().numpy()[0] == False] = 0
        # save mask region
        cv2.imwrite(os.path.join(output_dir, 'general_mask','%d/%d.jpg'%(idx,id)), mask_img.numpy())
        # save mask img
        cv2.imwrite(os.path.join(output_dir, 'general_img','%d/%d.jpg'%(idx,id)), img_filtered)

        # init local canvas
        min_x, max_x, min_y, max_y = find_bound_box(mask.cpu().numpy()[0])
        mask_img_cropped = mask_img[min_x:max_x,min_y:max_y]
        img_filtered_cropped = img_filtered[min_x:max_x,min_y:max_y]
        # save mask region
        cv2.imwrite(os.path.join(output_dir, 'local_mask','%d/%d.jpg'%(idx,id)), mask_img_cropped.numpy())
        # save mask img
        cv2.imwrite(os.path.join(output_dir, 'local_img','%d/%d.jpg'%(idx,id)), img_filtered_cropped)

    # json_data = {
    #     # 'tags_chinese': tags_chinese,
    #     'mask':[{
    #         'value': value,
    #         'label': 'background'
    #     }]
    # }
    # with open(os.path.join(output_dir, 'label.json'), 'w') as f:
    #     json.dump(json_data, f)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
        }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=False, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=False, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=False, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    # text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # get large tag catagory
    text_prompt = ''
    with open('ram_tag_list.txt','r') as f:
        for line in f:
            text_prompt += line.strip('\n').split('.')[0]
            text_prompt += '. '

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    
    # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # build loop
    image_paths = glob.glob('image_dataset' + '/*.jpg')

    # make folders
    for f in range(30):
        os.makedirs('%s/general_mask/%d'%(output_dir,f),exist_ok=True)
        os.makedirs('%s/local_mask/%d'%(output_dir,f),exist_ok=True)
        os.makedirs('%s/general_img/%d'%(output_dir,f),exist_ok=True)
        os.makedirs('%s/local_img/%d'%(output_dir,f),exist_ok=True)

    for idx,image_path in enumerate(image_paths):

        # load image 
        image_pil, image = load_image(image_path)

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,)

        parse_mask_region(image, output_dir, masks, idx)

        # draw output image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)

        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(output_dir, "grounded_sam_%d.jpg"%(idx)),
        #     bbox_inches="tight", dpi=300, pad_inches=0.0)

        # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
