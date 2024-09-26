import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import glob

# shapely Geo Process
# from shapely.geometry import Polygon,MultiPolygon,LineString,LinearRing,MultiPoint,MultiLineString#,Point,box

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor,
    SamAutomaticMaskGenerator
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram import inference_ram
from ram.models import ram, ram_plus
import torchvision.transforms as TS


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


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
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
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


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


# sel duplicated masks with defined max mask numbers
def mix_masks(SAM_mask,RAM_mask,num_limit,count_threshold,percent_threshold):

    masks_mixed = []
    mask_mixed_size = []
    masks_sorted = []
    valid_R = np.zeros(len(RAM_mask))

    for idx_S, mask_S in enumerate(SAM_mask):

        mask_S = mask_S.cpu().numpy()[0]
        count_S = len((mask_S == True)[0])
        valid_S = 0

        for idx_R, mask_R in enumerate(RAM_mask):

            mask_R = mask_R.cpu().numpy()[0]
            count_R = len((mask_R == True)[0])

            # find mask intersection count and percentage
            count_Inter = np.argwhere(np.logical_and(mask_S == True, mask_R == True)).shape[0]
            percent_S, percent_R = count_Inter/count_S, count_Inter/count_R

            if count_Inter > count_threshold: 
                if percent_S > percent_threshold or percent_R > percent_threshold:
                    mixed_mask = np.logical_or(mask_S == True, mask_R == True)
                    masks_mixed.append(mixed_mask)
                    mask_mixed_size.append(len((mixed_mask == True)[0]))
                    valid_S = 1
                    valid_R[idx_R] = 1
        
        # no closer RAM masks
        if valid_S == 0: masks_mixed.append(mask_S)

    # get remained RAM masks
    for idx_R in range(count_R): 
        if valid_R[idx_R] == 0: masks_mixed.append(RAM_mask[idx_R])
    
    # sort from large mask to small
    size_list = np.array(mask_mixed_size)
    sorted_list_crop = np.argsort(size_list[::-1])[:num_limit]
    for m in range(num_limit): masks_sorted.append(masks_mixed[sorted_list_crop[m]])

    return masks_sorted


def parse_mask_region(img, output_dir, mask_list, id):

    mask_img_all = img.copy()

    for idx, mask in enumerate(mask_list):

        # init general canvas
        mask_img = torch.zeros(mask_list.shape[-2:])
        mask_img[mask.cpu().numpy()[0] == True] = 255
        mask_img_all[mask.cpu().numpy()[0] == True] = seg_colors[idx]
        img_filtered = img.copy()
        img_filtered[mask.cpu().numpy()[0] == False,:] = 0

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

    cv2.imwrite(os.path.join(output_dir,'%d.jpg'%(id)), mask_img_all)


    # json_data = {
    #     # 'tags_chinese': tags_chinese,
    #     'mask':[{
    #         'value': value,
    #         'label': 'background'
    #     }]
    # }
    # with open(os.path.join(output_dir, 'label.json'), 'w') as f:
    #     json.dump(json_data, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_ram_plus", action="store_true", help="using ram plus"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument(
        "--image_path", default='D:/COCO/train2014',type=str, help="path to image file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument(
        "--max_seg", type=int, default=64, help="max number of segments per img"
    )
    parser.add_argument("--box_threshold", type=float, default=0.2, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.3, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--percent_threshold", type=float, default=0.3, help="percent threshold")
    parser.add_argument("--count_threshold", type=int, default=100, help="count threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    use_ram_plus = args.use_ram_plus
    image_path = args.image_path
    output_dir = args.output_dir
    max_seg = args.max_seg
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    percent_threshold = args.percent_threshold
    count_threshold = args.count_threshold
    device = args.device
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load SAM model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # initialize RAM model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([TS.Resize((384, 384)),TS.ToTensor(), normalize])
    
    # load RAM model
    if use_ram_plus: ram_model = ram_plus(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
    else: ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')

    ram_model.eval()
    ram_model = ram_model.to(device)

    # initialize SAM
    if use_sam_hq: predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else: predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    # initialize automated SAM
    auto_predictor = SamAutomaticMaskGenerator(
            model=build_sam(checkpoint=sam_checkpoint).to(device),
            points_per_side=int(max_seg/2), # 32
            min_mask_region_area=100000, # None
            pred_iou_thresh=0.95, # 0.88
            stability_score_thresh= 0.95, #0.95
            stability_score_offset = 1.0, 
            box_nms_thresh = 0.9, #0.7
            crop_n_layers = 0, 
            crop_nms_thresh = 0.9, #0.7
            crop_overlap_ratio = 512 / 1500,
            crop_n_points_downscale_factor = 1 
            ) 

    # load imgs paths
    image_paths = glob.glob(image_path + '/*.jpg')

    # seg colors, with default 64 max segments
    seg_colors = np.zeros((max_seg,3))

    # change if max_seg > 64
    for c in range(max_seg):
        seg_colors[c,0] = 255 - ((c+1) % 4)*63
        seg_colors[c,1] = 255 - (((c+1) // 4) % 4)*63
        seg_colors[c,2] = 255 - (((c+1) // 16) % 4)*63

    # make folders
    for f in range(max_seg):

        # entire img
        os.makedirs('%s/general_mask/%d'%(output_dir,f),exist_ok=True)

        # cropped mask
        os.makedirs('%s/local_mask/%d'%(output_dir,f),exist_ok=True)

        # entire img
        os.makedirs('%s/general_img/%d'%(output_dir,f),exist_ok=True)

        # cropped img
        os.makedirs('%s/local_img/%d'%(output_dir,f),exist_ok=True)

    # ram results
    os.makedirs('%s/ram'%(output_dir),exist_ok=True)

    # segment results
    os.makedirs('%s/segement'%(output_dir),exist_ok=True)

    mask_num = np.zeros(len(image_paths)) 

    for idxs,image_path in enumerate(image_paths[:100]):

        # load image
        image_pil, image = load_image(image_path)
        # visualize raw image
        # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
        raw_image = image_pil.resize((384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(device)

        # RAM Inference
        res = inference_ram(raw_image , ram_model)

        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        tags = res[0].replace(' |', ',')

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model, image, tags, box_threshold, text_threshold, device=device)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # auto SAM inference
        masks_autoSAM = auto_predictor.generate(image)

        # SAM with RAM inference
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        # print(f"After NMS: {boxes_filt.shape[0]} boxes")

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks_RAM, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )

        SAM_mask = [m['segmentation'] for m in masks_autoSAM]
        RAM_mask = [m['segmentation'] for m in masks_RAM]

        masks_filtered = mix_masks(SAM_mask,RAM_mask,max_seg,count_threshold,percent_threshold)

        parse_mask_region(image, output_dir, masks_filtered, idxs)

        if idxs % 100 == 0: print(idxs)

        # get mask counts
        # mask_num[idxs] = len(masks)

        '''
        
        
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        plt.title('RAM-tags' + tags + '\n')# + 'RAM-tags_chineseing: ' + tags_chinese + '\n'
        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, "ram","label_%d.jpg"%(idx)), 
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        '''
        # save_mask_data(output_dir, masks, boxes_filt, pred_phrases, idx)#tags_chinese

# python DataProcess_DynaCLIP.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --ram_checkpoint ram_swin_large_14m.pth --grounded_checkpoint groundingdino_swint_ogc.pth --sam_checkpoint sam_vit_h_4b8939.pth --output_dir "outputs_DynaCLIP" --device "cuda"