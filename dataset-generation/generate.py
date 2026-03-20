import argparse
import os
import numpy as np
import json
import torch
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
from tqdm import tqdm
# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS

import base64
from io import BytesIO

from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor, CLIPModel
import pycocotools.mask as mask_util
import webdataset as wds


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def encode_tensor_as_string(arr):
    if type(arr) != np.ndarray:
        arr = arr.data.cpu().numpy()
    return base64.b64encode(arr.tobytes()).decode('utf-8')


# load BLIP and CLIP model
def load_blip_clip():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    blip_model, blip_vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    )
    text_version = "/home/haoli84/.cache/huggingface/transformers"
    clip_text_model = CLIPModel.from_pretrained(text_version).cuda().eval()
    clip_text_processor = CLIPProcessor.from_pretrained(text_version)
    return blip_model, blip_vis_processors, clip_text_model, clip_text_processor


# preprocess text using CLIP
def preprocess_text(processor, input):
    if input == None:
        return None
    inputs = processor(text=input, return_tensors="pt", padding=True)
    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['pixel_values'] = torch.ones(1, 3, 224, 224).cuda()  # placeholder
    inputs['attention_mask'] = inputs['attention_mask'].cuda()
    return inputs


# get CLIP embeddings
def get_clip_feature_text(model, processor, input):
    which_layer_text = 'before'
    inputs = preprocess_text(processor, input)
    if inputs == None:
        return None
    outputs = model(**inputs)
    if which_layer_text == 'before':
        feature = outputs.text_model_output.pooler_output
        return feature


# generate caption using BLIP and get the CLIP embeddings
def forward_blipv2(raw_image, category_name, bbox, blip_model, blip_vis_processors, clip_text_model,
                   clip_text_processor):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area >= 32 * 32:
        raw_image_cropped = raw_image.crop(bbox)
        raw_image_cropped = blip_vis_processors["eval"](raw_image_cropped).unsqueeze(0).to(device)
        captions = blip_model.generate({"image": raw_image_cropped})
        instance_caption = captions[0]
        instance_caption_all = captions[0]

        if category_name != None and category_name != '':
            if category_name.lower() not in instance_caption.lower():
                instance_caption_all = category_name + '. ' + instance_caption
        blip_clip_embeddings = encode_tensor_as_string(
            get_clip_feature_text(clip_text_model, clip_text_processor, instance_caption_all))
        text_embedding_before = encode_tensor_as_string(
            get_clip_feature_text(clip_text_model, clip_text_processor, category_name))

    else:
        instance_caption = category_name
        blip_clip_embeddings = None
        text_embedding_before = encode_tensor_as_string(
            get_clip_feature_text(clip_text_model, clip_text_processor, category_name))

    return instance_caption, blip_clip_embeddings, text_embedding_before


# decode base64 to pillow image
def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')


# read images
def read_image_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    img = decode_base64_to_pillow(data['image'])
    return img, data


# apply image transformation
def apply_img_transform(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
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
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def is_all_zeros(tensor):
    return torch.all(tensor == 0)
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def mask_2_rle(binary_mask):
    rle = mask_util.encode(np.array(binary_mask[..., None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def save_mask_data(output_dir, mask_list, box_list, label_list, file_name, image_pil, output, blip_model,
                   blip_vis_processors, clip_text_model, clip_text_processor):
    value = 0

    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]

        mask = mask_list[value - 1].cpu().numpy()[0] == True
        rle = mask_2_rle(mask)

        box_xywh = [int(x) for x in box.numpy().tolist()]
        box_xywh[2] = box_xywh[2] - box_xywh[0]
        box_xywh[3] = box_xywh[3] - box_xywh[1]

        anno = get_base_anno_dict(is_stuff=0, is_thing=1, bbox=box_xywh, pred_score=float(logit), mask_value=value,
                                  rle=rle, category_name=name, area=box_xywh[-1] * box_xywh[-2])
        RGB_image = image_pil.convert('RGB')
        x1y1x2y2 = [int(x) for x in box.numpy().tolist()]
        instance_caption, blip_clip_embeddings, text_embedding_before = forward_blipv2(RGB_image, name, x1y1x2y2,
                                                                                       blip_model, blip_vis_processors,
                                                                                       clip_text_model,
                                                                                       clip_text_processor)
        anno['text_embedding_before'] = text_embedding_before
        if blip_clip_embeddings != None:
            anno['caption'] = instance_caption
            anno['blip_clip_embeddings'] = blip_clip_embeddings
        output['annos'].append(anno)

    with open(os.path.join(output_dir, 'label_{}.json'.format(file_name)), 'w') as f:
        json.dump(output, f)
        print("Saved {}/label_{}.json".format(output_dir, file_name))


# convert PIL image to base64
def encode_pillow_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_base_output_dict(image, dataset_name, image_path, data=None):
    output = {}
    if data != None:
        if 'similarity' in data:
            output['similarity'] = data['similarity']
        if 'AESTHETIC_SCORE' in data:
            output['AESTHETIC_SCORE'] = data['AESTHETIC_SCORE']
        if 'caption' in data:
            output['caption'] = data['caption']
        if 'width' in data:
            output['width'] = data['width']
        if 'height' in data:
            output['height'] = data['height']
        if 'file_name' in data:
            output['file_name'] = data['file_name']
        if 'is_det' in data:
            output['is_det'] = data['is_det']
        else:
            output['is_det'] = 0
        if 'image' in data:
            output['image'] = data['image']
    else:
        output['file_name'] = image_path
        output['is_det'] = 1
        output['image'] = encode_pillow_to_base64(image.convert('RGB'))
    output['dataset_name'] = dataset_name
    output['data_id'] = 1
    output['annos'] = []
    return output


def get_base_anno_dict(is_stuff, is_thing, bbox, pred_score, mask_value, rle, category_name, area):
    anno = {
        "id": 0,
        "isfake": 0,
        "isreflected": 0,
        "bbox": bbox,
        "mask_value": mask_value,
        "mask": rle,
        "pred_score": pred_score,
        "category_id": 0,
        "data_id": 0,
        "category_name": category_name,
        "text_embedding_before": "",
        "caption": "",
        "blip_clip_embeddings": "",
        "is_stuff": is_stuff,
        "is_thing": is_thing,
        "area": area
    }
    return anno


def get_args_parser():
    parser = argparse.ArgumentParser('Caption Generation script', add_help=False)
    parser.add_argument("--job_index", type=int, default=0, help="")
    parser.add_argument("--num_jobs", type=int, default=1, help="")
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
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="path to the json file with image path and image caption")
    parser.add_argument("--annotation_path", type=str,
                        help="path to the json file with annotation of source dataset")

    return parser

#函数。读取coco数据集的json格式标签文件
def read_coco_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_image_to_id_dict(json_path):
    # 加载COCO数据集的JSON文件
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 初始化字典，键是图片文件名，值是图片ID
    image_id_dict = {}

    # 遍历JSON文件中的images，查找与给定目录中的图片匹配的图片
    for image in coco_data['images']:
        file_name = image['file_name']
        image_id = image['id']

        # 获取图片文件名（不包含路径）
        image_key = os.path.splitext(os.path.basename(file_name))[0]

        # 将图片文件名和ID添加到字典
        image_id_dict[image_key] = image_id

    return image_id_dict

def create_id_to_category_dict(json_path):
    # 加载COCO数据集的JSON文件
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 初始化字典，键是图片文件名，值是图片ID
    image_id_dict = {}

    # 遍历JSON文件中的images，查找与给定目录中的图片匹配的图片
    for categories in coco_data['categories']:
        name = categories['name']
        id = categories['id']

        # 将图片文件名和ID添加到字典
        image_id_dict[id] = name

    return image_id_dict

## 函数。根据json标注返回image_id_to_annotations
def get_image_id_to_annotations(json_path):
    data = read_coco_json(json_path)
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    image_id_to_annotations = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        if image_id in image_id_to_annotations:
            image_id_to_annotations[image_id].append(annotation)
    return image_id_to_annotations

def main(args):
    # cfg
    config_file = args.config
    ram_checkpoint = args.ram_checkpoint
    grounded_checkpoint = args.grounded_checkpoint
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device

    # load model
    if args.annotation_path is None:
        model = load_model(config_file, grounded_checkpoint, device=device)

    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(), normalize
    ])

    # load model
    # use recognize_anything.ram if you are using Grounded-SAM commit before 2023-11-23
    # ram_model = recognize_anything.ram(pretrained=ram_checkpoint,
    ram_model = ram(pretrained=ram_checkpoint,
                    image_size=384,
                    vit='swin_l')
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()
    ram_model = ram_model.to(device)

    # load BLIP and CLIP model
    print("Initialize BLIP and CLIP model")
    blip_model, blip_vis_processors, clip_text_model, clip_text_processor = load_blip_clip()

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load images for model inference
    num_images = 0
    dataset_name = 'ss-sim-0.3-aesthetic-5'
    # read image and captions from json file
    json_path = args.train_data_path
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_paths = []
    image_captions = []
    for i in range(len(data)):
        image_paths.append(data[i]['image'])
        image_captions.append(data[i]['caption'])
    print(image_paths)

    # split the image list into multiple jobs
    n_images = len(image_paths)
    print('Overall number of images: ', n_images)
    num_imgs_per_job = n_images // args.num_jobs + 1
    start_idx = args.job_index * num_imgs_per_job
    end_idx = min((args.job_index + 1) * num_imgs_per_job, n_images)
    print("start_idx, end_idx", start_idx, end_idx)

    if args.annotation_path:
        # annotations = read_coco_json(args.annotation_path)
        image_to_id_dict = create_image_to_id_dict(args.annotation_path)
        image_id_to_annotations = get_image_id_to_annotations(args.annotation_path)
        id_to_category_dict = create_id_to_category_dict(args.annotation_path)


    # iterate over all images
    for image_path, image_caption in tqdm(zip(image_paths[start_idx:end_idx], image_captions[start_idx:end_idx])):
        img_meta_data = {}  # store image meta data
        # dataset = wds.WebDataset([tar]).decode("pil")
        img_name_base = image_path.split("/")[-1].split(".")[0]

        # read image and convert to RGB image using PIL
        image_pil = Image.open(image_path).convert("RGB")  # load image

        # save raw image
        img_meta_data['image'] = encode_pillow_to_base64(image_pil.convert('RGB'))

        # save the image caption
        img_meta_data['caption'] = image_caption

        # save file name
        file_name = img_name_base
        img_meta_data['file_name'] = image_path

        # get base output dictionary
        output = get_base_output_dict(image_pil, dataset_name, file_name, data=img_meta_data)
        image = apply_img_transform(image_pil)
        masks = 0
        # # visualize raw image
        # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

        if args.annotation_path is not None:
            if os.path.splitext(os.path.basename(image_path))[0] not in image_to_id_dict:
                continue
            img_id = image_to_id_dict[os.path.splitext(os.path.basename(image_path))[0]]
            # coco_annotations = read_coco_json(args.annotation_path)
            #根据文件路径获得对应图片id
            if img_id not in image_id_to_annotations:
                continue
            # if img_id < 450:
            #     continue

            annotations = image_id_to_annotations[img_id]
            boxes_filt = []
            scores = []
            pred_phrases = []
            masks = torch.zeros(len(annotations), 1, image_pil.height, image_pil.width)
            for index, annotation in enumerate(annotations):
                boxes_filt.append([annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0]+annotation['bbox'][2], annotation['bbox'][1]+annotation['bbox'][3]])
                scores.append(1.0)
                pred_phrases.append(id_to_category_dict[annotation['category_id']] + '(1)')
                #如果arg.train_data_path参数包含字符cityscapes
                if 'cityscapes' in args.train_data_path and 'segmentation' in annotation:
                    rle_mask = annotation['segmentation']
                    pixel_mask = coco_mask.decode(rle_mask)
                    mask_tensor = torch.from_numpy(pixel_mask.astype(np.uint8))
                    masks[index, 0, :, :] = mask_tensor

            boxes_filt = torch.tensor(boxes_filt)
            scores = torch.tensor(scores)

            # image = np.array(image_pil)
            # predictor.set_image(image)

            # use NMS to handle overlapped boxes
            # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
            # nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            # boxes_filt = boxes_filt[nms_idx]
            # pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        else:
            # run RAM model
            raw_image = image_pil.resize((384, 384))
            raw_image = transform(raw_image).unsqueeze(0).to(device)
            # use inference_ram.inference if you are using Grounded-SAM commit before 2023-11-23
            res = inference_ram(raw_image, ram_model)

            # Currently ", " is better for detecting single tags
            # while ". " is a little worse in some case
            tags = res[0].replace(' |', ',')

            # run grounding dino model
            boxes_filt, scores, pred_phrases = get_grounding_output(
                model, image, tags, box_threshold, text_threshold, device=device
            )
            image = np.array(image_pil)
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

        if type(masks)==int or (isinstance(masks, torch.Tensor) and torch.all(masks == 0)):
            image = np.array(image_pil)
            predictor.set_image(image)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
            try:
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(device),
                    multimask_output=False,
                )
            except:
                continue

        num_images += 1

        # save mask data
        save_mask_data(output_dir, masks, boxes_filt, pred_phrases, file_name, image_pil, output, blip_model,
                       blip_vis_processors, clip_text_model, clip_text_processor)
        # print("Processed {} image; {}".format(num_images, file_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Traing data generate script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)