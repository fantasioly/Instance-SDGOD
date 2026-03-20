import json
import random
import os
import json
import torch
import argparse
import numpy as np
import uuid
from functools import partial
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tkinter.messagebox import NO
from diffusers.utils import load_image
from diffusers import StableDiffusionXLImg2ImgPipeline

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.plms_instance import PLMSSamplerInst
from dataset.decode_item import sample_random_points_from_mask, sample_sparse_points_from_mask, decodeToBinaryMask, \
    reorder_scribbles

from skimage.transform import resize
from utils.checkpoint import load_model_ckpt
from utils.input import convert_points, prepare_batch, prepare_instance_meta
from utils.model import create_clip_pretrain_model, set_alpha_scale, alpha_generator

device = "cuda"


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1, max_objs)
    if has_mask == None:
        return mask

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0, idx] = value
        return mask


@torch.no_grad()
def get_model_inputs(meta, model, text_encoder, diffusion, clip_model, clip_processor, config,
                     grounding_tokenizer_input, starting_noise=None, instance_input=False):
    if not instance_input:
        # update config from args
        config.update(vars(args))
        config = OmegaConf.create(config)

    # prepare a batch of samples
    batch = prepare_batch(meta, batch=config.num_images, max_objs=30, model=clip_model, processor=clip_processor,
                          image_size=model.image_size, use_masked_att=True, device="cuda")
    context = text_encoder.encode([meta["prompt"]] * config.num_images)

    # unconditional input
    if not instance_input:
        uc = text_encoder.encode(config.num_images * [""])
        if args.negative_prompt is not None:
            uc = text_encoder.encode(config.num_images * [args.negative_prompt])
    else:
        uc = None

    # sampler
    if not instance_input:
        alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
        if config.mis > 0:
            sampler = PLMSSamplerInst(diffusion, model, alpha_generator_func=alpha_generator_func,
                                      set_alpha_scale=set_alpha_scale, mis=config.mis)
        else:
            sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                                  set_alpha_scale=set_alpha_scale)
        steps = 50
    else:
        sampler, steps = None, None

    # grounding input
    grounding_input = grounding_tokenizer_input.prepare(batch, return_att_masks=return_att_masks)

    # model inputs
    input = dict(x=starting_noise, timesteps=None, context=context, grounding_input=grounding_input)
    return input, sampler, steps, uc, config


@torch.no_grad()
def run(meta, model, autoencoder, text_encoder, diffusion, clip_model, clip_processor, config,
        grounding_tokenizer_input, starting_noise=None, guidance_scale=None):
    # prepare models inputs
    input, sampler, steps, uc, config = get_model_inputs(meta, model, text_encoder, diffusion, clip_model,
                                                         clip_processor, config, grounding_tokenizer_input,
                                                         starting_noise, instance_input=False)
    if guidance_scale is not None:
        config.guidance_scale = guidance_scale

    # prepare models inputs for each instance if MIS is used
    if args.mis > 0:
        input_all = [input]
        for i in range(len(meta['phrases'])):
            meta_instance = prepare_instance_meta(meta, i)
            input_instance, _, _, _, _ = get_model_inputs(meta_instance, model, text_encoder, diffusion, clip_model,
                                                          clip_processor, config, grounding_tokenizer_input,
                                                          starting_noise, instance_input=True)
            input_all.append(input_instance)
    else:
        input_all = input

    # start sampling
    shape = (config.num_images, model.in_channels, model.image_size, model.image_size)
    with torch.autocast(device_type=device, dtype=torch.float16):
        samples_fake = sampler.sample(S=steps, shape=shape, input=input_all, uc=uc,
                                      guidance_scale=config.guidance_scale)
    samples_fake = autoencoder.decode(samples_fake)

    # define output folder
    output_folder = os.path.join(args.output, meta["save_folder_name"])
    # os.makedirs(output_folder, exist_ok=True)

    # start = len(os.listdir(output_folder))
    # image_ids = list(range(start, start + config.num_images))
    # print(image_ids)

    # visualize the boudning boxes
    # image_boxes = draw_boxes(meta["locations"], meta["phrases"],
    #                          meta["prompt"] + ";alpha=" + str(meta['alpha_type'][0]))
    # img_name = os.path.join(output_folder, str(image_ids[0]) + '_boxes.png')
    # image_boxes.save(img_name)
    # print("saved image with boxes at {}".format(img_name))

    # if use cascade model, we will use SDXL-Refiner to refine the generated images
    if config.cascade_strength > 0:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "/home/haoli84/.cache/huggingface/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True
        )
        pipe = pipe.to("cuda:0")
        strength, steps = config.cascade_strength, 20  # default setting, need to be manually tuned.

    # save the generated images
    image_sample = []
    refined_images = []
    for sample in samples_fake:
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
        sample = Image.fromarray(sample.astype(np.uint8))
        if config.cascade_strength > 0:
            prompt = meta["prompt"]
            refined_image = pipe(prompt, image=sample, strength=strength, num_inference_steps=steps).images[0]
            refined_images.append(refined_image)
            # refined_image.save(
            #     os.path.join(output_folder, img_name.replace('.png', '_xl_s{}_n{}.png'.format(strength, steps))))
        image_sample.append(sample)
    if config.cascade_strength > 0:
        return refined_images
    else:
        return image_sample

def rescale_box(bbox, width, height):
    x0 = bbox[0] / width
    y0 = bbox[1] / height
    x1 = (bbox[0] + bbox[2]) / width
    y1 = (bbox[1] + bbox[3]) / height
    return [x0, y0, x1, y1]


def get_point_from_box(bbox):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]


def rescale_points(point, width, height):
    return [point[0] / float(width), point[1] / float(height)]


def rescale_scribbles(scribbles, width, height):
    return [[scribble[0] / float(width), scribble[1] / float(height)] for scribble in scribbles]


# draw boxes given a lits of boxes: [[top left cornor, top right cornor, width, height],]
# show descriptions per box if descriptions is not None
def draw_boxes(boxes, descriptions=None, caption=None):
    width, height = 512, 512
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    boxes = [[int(x * width) for x in box] for box in boxes]
    for i, box in enumerate(boxes):
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=(0, 0, 0), width=2)
    if descriptions is not None:
        for idx, box in enumerate(boxes):
            draw.text((box[0], box[1]), descriptions[idx], fill="black")
    if caption is not None:
        draw.text((0, 0), caption, fill=(255, 102, 102))
    return image

domain_pre_prompt = {
    "daytimeclear": "a driving footage photo of a street during daytime",
    "duskrainy": "a driving footage photo of a rainy street during dusk",
    "nightsunny": "a driving footage photo of a dark street during night",
    "nightrainy": "a driving footage photo of a rainy dark street during night",
    "daytimefoggy": "a driving footage photo of a foggy street during daytime",
}

domian_attribute_dict = {
    "scene": ["urban scene", "street scene", "city street", "cityscape"],
    "action": ["driving", "moving", "walking", "parked"],
    "daytimeclear": {
        "attribute": ["daytime"],
        "weather": ["overcast", "cloudy", "clear"],
        "time":  ["daytime", "morning", "afternoon"],
        "scene": ["highway scene", "street scene", "residential scene"],
    },
    "duskrainy": {
        "attribute": ["dusk", "raining", "raindrop"],
        "weather": ["raining", "heavy rain"],
        "time": ["dusk"],
        "scene": ["raindrop highway scene", "raindrop street scene", "raindrop city street"],
    },
    "nightsunny": {
        "attribute": ["night", "dark", " black sky"],
        "weather": ["clear", "cloudy"],
        "time": ["night"],
        "scene": ["dark highway", "dark cityscape scene", "dark street"],
    },
    "nightrainy": {
        "attribute": ["night", "rain", "raindrop", "darkness"],
        "weather": ["rainy", "storm", "heavy rain"],
        "time": ["night"],
        "scene": ["raindrop residential", "raindrop street", "raindrop highway"],
    },

    "daytimefoggy": {
        "attribute": ["foggy","blurry"],
        "weather": ["fog"],
        "time": ["daytime"],
        "scene": ["foggy urban scene", "foggy street scene", "foggy city street", "foggy cityscape"],
    }
}

bbox_attribute_dict = {
    "car": {
        "object": ["car", "sedan", "SUV", "black car", "white car", "gray car"],
        "action": ["moving", "driving", "parked"]
    },
    "truck": {
        "object": ["truck"],
        "action": ["moving", "driving", "parked"]
    },
    "person": {
        "object": ["pedestrian", "people", "person", "walker"],
        "action": ["walking", "running", "standing"]
    },
    "bike": {
        "object": ["mountain bike", "bicycle", "bike"],
        "action": ["moving", "parked"]
    },
    "bus": {
        "object": ["bus", "city bus"],
        "action": ["moving", "driving", "parked"]
    },
    "motor": {
        "object": ["motorcycle", "motorbike"],
        "action": ["moving", "driving", "parked"]
    },
    "rider": {
        "object": ["cyclist", "motorcyclist", "motor scooter", "rider", "motorcycle", "motorist"],
        "action": ["moving", "riding", "standing"]
    },
}

dict_category_to_id = {
    "bus": 1,
    "bike": 2,
    "car": 3,
    "motor": 4,
    "person": 5,
    "rider": 6,
    "truck": 7,
}
annotations = []
images = []
categories = [{
    "id": 1,
    "name": "bus",
    "supercategory": "none"
},
    {
        "id": 2,
        "name": "bike",
        "supercategory": "none"
    },
    {
        "id": 3,
        "name": "car",
        "supercategory": "none"
    },
    {
        "id": 4,
        "name": "motor",
        "supercategory": "none"
    },
    {
        "id": 5,
        "name": "person",
        "supercategory": "none"
    },
    {
        "id": 6,
        "name": "rider",
        "supercategory": "none"
    },
    {
        "id": 7,
        "name": "truck",
        "supercategory": "none"
    }]

def get_category_id(annotations, category_name):
    """
    Find the category ID for a given category name in the annotations.

    Parameters:
    - annotations: the 'annotations' dictionary loaded from the COCO JSON
    - category_name: the name of the category to find the ID for

    Returns:
    - The category ID as an integer
    """
    categories = annotations['categories']
    for category in categories:
        if category['name'] == category_name:
            return category['id']
    return None

def load_coco_annotations(anno_file, category_name):
    """
    Load COCO annotations from a JSON file and extract bboxes for a given category.

    Parameters:
    - anno_file: path to the COCO annotations JSON file
    - category_name: the category to extract bboxes for

    Returns:
    - A list of bboxes, each bbox is a list of [xmin, ymin, width, height]
    """
    with open(anno_file, 'r') as f:
        annotations = json.load(f)
    category_id = get_category_id(annotations, category_name)
    # Filter annotations to get only the specified category
    category_bboxes = [
        anno['bbox'] for anno in annotations['annotations']
        if anno['category_id'] == category_id
    ]

    return category_bboxes

# Check if bbox is within image bounds. If not, translate bbox to a random position within the image.
# If bbox is within bounds, return original bbox.
def check_bbox_in_image(bbox, image_width, image_height):
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height

    if xmin < 0 or xmax > image_width or ymin < 0 or ymax > image_height:
        # Bbox is out of image bounds, move it to a random position within the image
        xmin = random.randint(0, image_width - width)
    return [xmin, ymin, width, height]


def generate_captions(domain,json_path, class_list, bboxes_list):
    ann = []

    for category_name in class_list:
        scene = random.choice(domian_attribute_dict[domain]["scene"])
        time = random.choice(domian_attribute_dict[domain]["time"])
        weather = random.choice(domian_attribute_dict[domain]["weather"])
        action = random.choice(bbox_attribute_dict[category_name]["action"])
        box_prompt = "A " + category_name + " is " + action + " in a " + weather + " " + scene + " during " + time + "."
        bbox = random.choice(bboxes_list[category_name])
        bbox = check_bbox_in_image(bbox, 720, 720)
        ann.append({
            "bbox": bbox,
            "mask": [],
            "category_name": category_name,
            "caption": box_prompt
        })

    scene = random.choice(domian_attribute_dict[domain]["scene"])
    time = random.choice(domian_attribute_dict[domain]["time"])
    weather = random.choice(domian_attribute_dict[domain]["weather"])
    image_caption_template = "A driving footage photo of an autonomous vehicle in a {scene} during {time} with {weather} weather conditions."
    image_caption =  image_caption_template.format(scene=scene, time=time, weather=weather)
    image_input_instdiff = {
        "caption": image_caption,
        "width": 720,
        "height": 720,
        "annos": ann,
    }
    return image_input_instdiff



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,  default="OUTPUT/rare_classes_genaration", help="root folder for output")
    parser.add_argument("--num_images", type=int, default=2, help="")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--input_json", type=str, default='demos/demo_cat_dog_robin.json', help="json files for instance-level conditions")
    parser.add_argument("--ckpt", type=str, default='pretrained/instancediffusion_sd15.pth', help="pretrained checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--alpha", type=float, default=1, help="the percentage of timesteps using grounding inputs")
    parser.add_argument("--mis", type=float, default=0.4, help="the percentage of timesteps using MIS")
    parser.add_argument("--cascade_strength", type=float, default=0, help="strength of SDXL Refiner.")
    parser.add_argument("--test_config", type=str, default="configs/test_mask.yaml", help="config for model inference.")
    parser.add_argument("--target_domains", type=list, default=["daytimefoggy","nightrainy","duskrainy","nightsunny","daytimeclear"])
    # parser.add_argument("--target_domains", type=list, default=["daytimefoggy"])
    parser.add_argument("--genetate_num", type=int, default=2)
    parser.add_argument("--json_path", type=str, default='/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear/voc07_train.json')

    args = parser.parse_args()

    return_att_masks = False
    ckpt = args.ckpt

    seed = args.seed
    save_folder_name = f"gc{args.guidance_scale}-seed{seed}-alpha{args.alpha}"
    # set seed
    torch.manual_seed(seed)
    starting_noise = torch.randn(args.num_images, 4, 64, 64).to(device)

    model, autoencoder, text_encoder, diffusion, config = load_model_ckpt(ckpt, args, device)
    clip_model, clip_processor = create_clip_pretrain_model()

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    bboxeslist = {}
    bboxeslist['person'] = load_coco_annotations(args.json_path, 'person')
    bboxeslist['bike'] = load_coco_annotations(args.json_path, 'bike')
    bboxeslist['motor'] = load_coco_annotations(args.json_path, 'motor')
    bboxeslist['bus'] = load_coco_annotations(args.json_path, 'bus')
    bboxeslist['rider'] = load_coco_annotations(args.json_path, 'rider')

    img_id = 0
    bbox_id = 0
    for domain in args.target_domains:
        for i in range(args.genetate_num):
            data = generate_captions(domain,args.json_path, ["rider", "bike", "motor", "person", "bus"], bboxeslist)

            # START: READ BOXES AND BINARY MASKS
            boxes = []
            binay_masks = []
            # class_names = []
            instance_captions = []
            points_list = []
            scribbles_list = []
            prompt = data['caption']
            crop_mask_image = False
            for inst_idx in range(len(data['annos'])):
                if "mask" not in data['annos'][inst_idx] or data['annos'][inst_idx]['mask'] == []:
                    instance_mask = np.zeros((512, 512, 1))
                else:
                    instance_mask = decodeToBinaryMask(data['annos'][inst_idx]['mask'])
                    if crop_mask_image:
                        # crop the instance_mask to 512x512, centered at the center of the instance_mask image
                        # get the center of the instance_mask
                        center = np.array([instance_mask.shape[0] // 2, instance_mask.shape[1] // 2])
                        # get the top left corner of the crop
                        top_left = center - np.array([256, 256])
                        # get the bottom right corner of the crop
                        bottom_right = center + np.array([256, 256])
                        # crop the instance_mask
                        instance_mask = instance_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
                        binay_masks.append(instance_mask)
                        data['width'] = 512
                        data['height'] = 512
                    else:
                        binay_masks.append(instance_mask)

                if "bbox" not in data['annos'][inst_idx]:
                    boxes.append([0, 0, 0, 0])
                else:
                    boxes.append(data['annos'][inst_idx]['bbox'])
                if 'point' in data['annos'][inst_idx]:
                    points_list.append(data['annos'][inst_idx]['point'])
                if "scribble" in data['annos'][inst_idx]:
                    scribbles_list.append(data['annos'][inst_idx]['scribble'])
                # class_names.append(data['annos'][inst_idx]['category_name'])
                instance_captions.append(data['annos'][inst_idx]['caption'])
                # show_binary_mask(binay_masks[inst_idx])

            # END: READ BOXES AND BINARY MASKS
            img_info = {}
            img_info['width'] = data['width']
            img_info['height'] = data['height']

            locations = [rescale_box(box, img_info['width'], img_info['height']) for box in boxes]
            phrases = instance_captions

            # get points for each instance, if not provided, use the center of the box
            if len(points_list) == 0:
                points = [get_point_from_box(box) for box in locations]
            else:
                points = [rescale_points(point, img_info['width'], img_info['height']) for point in points_list]

                # get binary masks for each instance, if not provided, use all zeros
            binay_masks = []
            for i in range(len(locations) - len(binay_masks)):
                binay_masks.append(np.zeros((512, 512, 1)))

            # get scribbles for each instance, if not provided, use random scribbles
            if len(scribbles_list) == 0:
                for idx, mask_fg in enumerate(binay_masks):
                    # get scribbles from segmentation if scribble is not provided
                    scribbles = sample_random_points_from_mask(mask_fg, 20)
                    scribbles = convert_points(scribbles, img_info)
                    scribbles_list.append(scribbles)
            else:
                scribbles_list = [rescale_scribbles(scribbles, img_info['width'], img_info['height']) for scribbles in
                                  scribbles_list]
                scribbles_list = reorder_scribbles(scribbles_list)

            print("num of inst captions, masks, boxes and points: ", len(phrases), len(binay_masks), len(locations),
                  len(points))

            # get polygons for each annotation
            polygons_list = []
            segs_list = []
            for idx, mask_fg in enumerate(binay_masks):
                # binary_mask = mask_fg[:,:,0]
                polygons = sample_sparse_points_from_mask(mask_fg, k=256)
                if polygons is None:
                    polygons = [0 for _ in range(256 * 2)]
                polygons = convert_points(polygons, img_info)
                polygons_list.append(polygons)

                segs_list.append(resize(mask_fg.astype(np.float32), (512, 512, 1)))

            segs = np.stack(segs_list).astype(np.float32).squeeze() if len(segs_list) > 0 else segs_list
            polygons = polygons_list
            scribbles = scribbles_list

            meta_list = [
                # grounding inputs for generation
                dict(
                    ckpt=ckpt,
                    prompt=prompt,
                    phrases=phrases,
                    polygons=polygons,
                    scribbles=scribbles,
                    segs=segs,
                    locations=locations,
                    points=points,
                    alpha_type=[args.alpha, 0.0, 1 - args.alpha],
                    save_folder_name=save_folder_name
                ),
            ]

            for meta in meta_list:
                image_sample = run(meta, model, autoencoder, text_encoder, diffusion, clip_model, clip_processor, config,
                    grounding_tokenizer_input, starting_noise, guidance_scale=args.guidance_scale)

            img_save_path = args.output+"/VOC2007/JPEGImages/"
            os.makedirs(img_save_path, exist_ok=True)
            # Save images and corresponding annotation files
            for sample in image_sample:
                img_name = domain + '_' + str(img_id) + '.png'
                sample.save(os.path.join(img_save_path, img_name))
                new_image = {
                    "file_name": img_name,
                    "id":img_id,
                    "height": 720,
                    "width": 720,
                    # You may need to add more fields here as required by your specific COCO dataset structure
                }
                images.append(new_image)

                for ann in data['annos']:
                # Define a new annotation
                    new_annotation = {
                        "area": ann['bbox'][2]*ann['bbox'][3],
                        "bbox": [int(x*512/720) for x in ann['bbox']],
                        "category_id": dict_category_to_id[ann['category_name']],  # This should match an existing category's id in your dataset
                        "image_id": img_id,
                        "id": bbox_id,
                        "iscrowd": 0,
                        "segmentation": []
                    }
                    annotations.append(new_annotation)
                    bbox_id += 1
                img_id += 1
        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
    with open(os.path.join(args.output,"train.json"), 'w') as f:
        json.dump(coco_data, f, indent=4)
