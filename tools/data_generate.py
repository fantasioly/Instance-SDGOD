# 用于生成数据集的脚本，输入源域数据集图片和标签、需要生成的目标域，输出目标域数据集图片和标签
import argparse
import numpy as np
import random

import torch

from PIL import Image, ImageDraw
from recognize_anything.ram.models import tag2text
from recognize_anything.ram import inference_tag2text as inference_tag2text
from recognize_anything.ram import inference_tag2text_insert
from recognize_anything.ram import get_transform

import os
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
import json
from tqdm import tqdm

from functools import partial
from omegaconf import OmegaConf
from diffusers import StableDiffusionXLImg2ImgPipeline

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.plms_instance import PLMSSamplerInst
from dataset.decode_item import sample_random_points_from_mask, sample_sparse_points_from_mask, decodeToBinaryMask, reorder_scribbles

from skimage.transform import resize
from utils.checkpoint import load_model_ckpt
from utils.input import convert_points, prepare_batch, prepare_instance_meta
from utils.model import create_clip_pretrain_model, set_alpha_scale, alpha_generator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 定义目标域属性词典，一共5个域，daytimeclear、duskrainy、nightsunny、nightrainy、daytimefoggy,
# 每个目标域属性对应一个字典，包含attribute、weather、time、scene和7个name类别：car、person、bicycle、motorcycle、truck、bus、train
domain_pre_prompt = {
    "daytimeclear": "a cityscape photo of a street during daytime",
    "duskrainy": "a cityscape photo of a rainy street during dusk",
    "nightsunny": "a cityscape photo of a dark street during night",
    "nightrainy": "a cityscape photo of a rainy dark street during night",
    "daytimefoggy": "a cityscape photo of a foggy street during daytime",
}

domian_attribute_dict = {
    "scene": ["urban scene", "street scene", "city street", "cityscape"],
    "action": ["driving", "moving", "walking", "parked"],
    "daytimeclear": {
        "attribute": ["daytime"],
        # "weather": ["morning light", "morning sun", "afternoon sun", "bright", "sunny"],
        "weather": ["overcast", "sunny"],
        "time":  ["daytime", "morning", "afternoon"],
        "scene": ["highway scene", "street scene", "cityscape scene"],
        "other": ["crowd", "crowded", "downtown", "freeway", "highway", "intersection", "neighbourhood"]
    },
    "duskrainy": {
        "attribute": ["dusk", "raining", "raindrop"],
        "weather": ["raining", "heavy rain"],
        "time": ["dusk"],
        "scene": ["raindrop highway scene", "raindrop street scene", "raindrop city street"],
        "other": ["crowd", "crowded", "highway", "intersection", "headlights", "stop light"]
    },
    "nightsunny": {
        "attribute": ["night", "dark", " black sky"],
        "weather": ["clear", "cloudy"],
        "time": ["night"],
        "scene": ["dark highway", "dark cityscape scene", "dark street"],
        "other": ["headlights", "stop light"]
    },
    "nightrainy": {
        "attribute": ["night", "rain", "raindrop", "darkness"],
        "weather": ["rainy", "storm", "heavy rain"],
        "time": ["night"],
        "scene": ["raindrop residential", "raindrop street", "raindrop highway"],
        "other": ["headlights", "stop light"]
    },
    # "daytimefoggy": {
    #     "attribute": "foggy",
    #     "weather": "fog, mist",
    #     "time": "daytime, morning, afternoon",
    #     "scene": "urban scene, street scene, city street, cityscape",
    #     "other": "crowd, crowded, downtown, freeway, highway, intersection, neighbourhood"
    # },
    "daytimefoggy": {
        "attribute": ["foggy","blurry"],
        "weather": ["fog"],
        "time": ["daytime"],
        "scene": ["foggy urban scene", "foggy street scene", "foggy city street", "foggy cityscape"],
        "other": ["crowd", "crowded", "downtown", "freeway", "highway", "intersection", "neighbourhood"]
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
        "object": ["motorcycle", "moped", "motorbike"],
        "action": ["moving", "driving", "parked"]
    },
    "rider": {
        "object": ["cyclist", "motorcyclist", "motor scooter", "rider", "motorcycle", "motorist"],
        "action": ["moving", "riding", "standing"]
    },
}

device = "cuda"



# 根据目标域、attribute_dict、类别，返回组合的属性列表
def get_region_list(target_domain, category, weather, scene, time):
    #object从列表bbox_attribute_dict[category]中随机选择一个
    object = random.choice(bbox_attribute_dict[category]["object"])
    action = random.choice(bbox_attribute_dict[category]["action"])
    # weather = random.choice(domian_attribute_dict[target_domain]["weather"])
    # scene = random.choice(domian_attribute_dict[target_domain]["scene"])
    # time = random.choice(domian_attribute_dict[target_domain]["time"])
    attribute_list = [object, action, weather, scene, time]
    if target_domain == "duskrainy" or target_domain == "nightrainy":
        region_prompt = "A " + object + " is " + action + " in a " + weather + " " + scene + " during " + time + ", " + "it is raining."
        # region_prompt = "A " + object + " is " + action + " in a " + weather + " " + scene + " during " + time + "."
    else:
        region_prompt = "A " + object + " is " + action + " in a " + weather + " " + scene + " during " + time + "."
    return attribute_list, region_prompt

# 2. 读取源域数据集图片和标签，生成目标域数据集图片和标签

#函数。读取coco数据集的json格式标签文件
def read_coco_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# 函数。读取源域数据集图片和标签
def read_coco_data(image_dir, json_path):
    data = read_coco_json(json_path)
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    image_id_to_annotations = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)

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


def filter_captions(target_domain, prompt, captions, key_words):
    # 计算prompt的特征向量
    vectorizer = CountVectorizer().fit([prompt])
    prompt_vector = vectorizer.transform([prompt])

    filtered_captions = []
    # 遍历每个caption
    for caption in captions:
        # 计算caption的特征向量
        caption_vector = vectorizer.transform([caption])

        # 计算caption与prompt之间的相似度（余弦相似度）
        similarity = cosine_similarity(prompt_vector, caption_vector)[0][0]

        # 检查caption是否包含所有必须的关键词汇之一
        contains_key_words = any(keyword in caption.lower() for keyword in key_words)
        # contains_key_words = all(keyword in caption.lower() for keyword in key_words)

        # 如果相似度高于阈值且包含必须的关键词汇，则将其保留
        if similarity > 0.6 and contains_key_words:
            filtered_captions.append((caption, similarity))

    # 根据相似度对captions进行排序
    if len(filtered_captions) > 1:
        filtered_captions.sort(key=lambda x: x[1], reverse=True)
        return [caption[0] for caption in filtered_captions][0]
    else:
        return domain_pre_prompt[target_domain]

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
        # steps = 60
    else:
        sampler, steps = None, None

    # grounding input
    grounding_input = grounding_tokenizer_input.prepare(batch, return_att_masks=return_att_masks)

    # model inputs
    input = dict(x=starting_noise, timesteps=None, context=context, grounding_input=grounding_input)
    return input, sampler, steps, uc, config


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

def generate_image(data, target_domain,filename,model, autoencoder, text_encoder, diffusion, config,clip_model, clip_processor, grounding_tokenizer_input):
    args = parser.parse_args()
    return_att_masks = False
    ckpt = args.ckpt

    seed = args.seed
    save_folder_name = f"gc{args.guidance_scale}-seed{seed}-alpha{args.alpha}"
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
            instance_mask = np.zeros((512,512,1))
        else:
            instance_mask = decodeToBinaryMask(data['annos'][inst_idx]['mask'])
            if crop_mask_image:
                # crop the instance_mask to 512x512, centered at the center of the instance_mask image
                # get the center of the instance_mask
                center = np.array([instance_mask.shape[0]//2, instance_mask.shape[1]//2])
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
            boxes.append([0,0,0,0])
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
        binay_masks.append(np.zeros((512,512,1)))

    # get scribbles for each instance, if not provided, use random scribbles
    if len(scribbles_list) == 0:
        for idx, mask_fg in enumerate(binay_masks):
            # get scribbles from segmentation if scribble is not provided
            scribbles = sample_random_points_from_mask(mask_fg, 20)
            scribbles = convert_points(scribbles, img_info)
            scribbles_list.append(scribbles)
    else:
        scribbles_list = [rescale_scribbles(scribbles, img_info['width'], img_info['height']) for scribbles in scribbles_list]
        scribbles_list = reorder_scribbles(scribbles_list)

    print("num of inst captions, masks, boxes and points: ", len(phrases), len(binay_masks), len(locations), len(points))

    # get polygons for each annotation
    polygons_list = []
    segs_list = []
    for idx, mask_fg in enumerate(binay_masks):
        # binary_mask = mask_fg[:,:,0]
        polygons = sample_sparse_points_from_mask(mask_fg, k=256)
        if polygons is None:
            polygons = [0 for _ in range(256*2)]
        polygons = convert_points(polygons, img_info)
        polygons_list.append(polygons)

        segs_list.append(resize(mask_fg.astype(np.float32), (512, 512, 1)))

    segs = np.stack(segs_list).astype(np.float32).squeeze() if len(segs_list) > 0 else segs_list
    polygons = polygons_list
    scribbles = scribbles_list

    meta_list = [
        # grounding inputs for generation
        dict(
            ckpt = ckpt,
            prompt = prompt,
            phrases = phrases,
            polygons = polygons,
            scribbles = scribbles,
            segs = segs,
            locations = locations,
            points = points,
            alpha_type = [args.alpha, 0.0, 1-args.alpha],
            save_folder_name=save_folder_name
        ),
    ]

    # set seed
    torch.manual_seed(seed)
    starting_noise = torch.randn(args.num_images, 4, 64, 64).to(device)

    for meta in meta_list:
        run_generate_image(meta, model, autoencoder, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise, guidance_scale=args.guidance_scale, target_domain = target_domain, filename=filename)

@torch.no_grad()
def run_generate_image(meta, model, autoencoder, text_encoder, diffusion, clip_model, clip_processor, config,
        grounding_tokenizer_input, starting_noise=None, guidance_scale=None, target_domain = None, filename=None):
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

    # output_folder = os.path.join(args.output + args.target_domain, meta["save_folder_name"])
    output_folder = os.path.join(args.output, args.target_domain)
    os.makedirs(output_folder, exist_ok=True)

    start = len(os.listdir(output_folder))
    image_ids = list(range(start, start + config.num_images))
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
    for image_id, sample in zip(image_ids, samples_fake):
        # img_name = str(int(image_id)) + '.png'
        img_name = filename
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        #如果sample的requires_grad是True，则改成False
        if sample.requires_grad:
            sample = sample.detach()
        sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
        sample = Image.fromarray(sample.astype(np.uint8))
        if config.cascade_strength > 0:
            prompt = meta["prompt"]
            refined_image = pipe(prompt, image=sample, strength=strength, num_inference_steps=steps).images[0]
            refined_image.save(
                os.path.join(output_folder, img_name.replace('.png', '_xl_s{}_n{}.png'.format(strength, steps))))

        # 将完整的图片路径拆分为文件夹路径和文件名
        img_folder, img_filename = os.path.split(img_name)
        # 检查并创建嵌套目录
        if not os.path.exists(os.path.join(output_folder, img_folder)):
            os.makedirs(os.path.join(output_folder, img_folder), exist_ok=True)

        sample.save(os.path.join(output_folder, img_name))
        # sample.save(os.path.join(output_folder, img_folder + "-" + str(image_id) + "-" + img_filename))
def generate_data(source_image_dir, source_json_path, target_domain, model, autoencoder, text_encoder, diffusion, config,clip_model, clip_processor, grounding_tokenizer_input,generate_prompt=False, max_bbox_num = 15):

    image_id_to_annotations = get_image_id_to_annotations(source_json_path)
    data = read_coco_json(source_json_path)
    images = data['images']
    categories = data['categories']
    categories_name_dict = {}
    for category in categories:
        categories_name_dict[category['id']] = category['name']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=384)

    tagmodel = tag2text(pretrained='/cpfs/user/haoli84/code/InstanceDiffusion/pretrained/tag2text_swin_14m.pth',
                             image_size=384,
                             vit='swin_b',
                             delete_tag_index=[127,2961, 3351, 3265, 3338, 3355, 3359])
    tagmodel.threshold = 0.68  # threshold for tagging
    tagmodel.eval()
    tagmodel = tagmodel.to(device)
    specified_tags = 'None'


    #对每张图片输入tag2text模型，得到每张图片的image caption
    for images_data in tqdm(images):

        image_id = images_data['id']
        image_path = os.path.join(source_image_dir, images_data['file_name'])
        height = images_data['height']
        width = images_data['width']
        image = transform(Image.open(image_path)).unsqueeze(0).to(device)

        # "annos"包含image_id_to_annotations[image_id]里每个标注的"bbox"、"mask"、"category_name"、"caption"
        ann = []
        #判断image_id_to_annotations是否有image_id，有则遍历image_id_to_annotations[image_id]
        if image_id not in image_id_to_annotations:
            continue
        if len(image_id_to_annotations[image_id])> max_bbox_num:
            continue

        # if image_id <12000:
        #     print("continue")
        #     continue

        image_tag = domian_attribute_dict[target_domain]['attribute']
        res = inference_tag2text_insert(image, tagmodel, image_tag, num_return_sequences=20)
        image_caption = res[2]
        image_caption = filter_captions(target_domain, res[1].replace(' | ', ', '), image_caption, image_tag)

        #当前图片的场景
        weather = random.choice(domian_attribute_dict[target_domain]["weather"])
        scene = random.choice(domian_attribute_dict[target_domain]["scene"])
        time = random.choice(domian_attribute_dict[target_domain]["time"])

        for annotation in image_id_to_annotations[image_id]:
            #bbox为[x_min.y_min,width,height]
            bbox = annotation['bbox']
            if bbox == [0, 0, 0, 0]:
                continue
            mask = annotation['segmentation']
            category_id = annotation['category_id']
            category_name = categories_name_dict[category_id]
            # box_tag = get_category_caption(category_name, target_domain)
            box_tag_list, box_prompt = get_region_list(target_domain,category_name, weather, scene, time)

            if generate_prompt :
            #使用tag2text模型生成box_caption
                region = Image.open(image_path)
                region = region.crop((bbox[0],bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                region = transform(region).unsqueeze(0).to(device)
                box_caption = tagmodel.text_generate(region, tag_input = box_tag_list, sample = True)
            else:
            #使用模板生成box_caption
                box_caption = box_prompt
            # box_caption = tagmodel.generate(region)

            ann.append({
                "bbox": bbox,
                "mask": [],
                "category_name": category_name,
                "caption": box_caption
            })


        #构建新的json，作为instansdiffusion的输入，包含“caption”、"width"、"height"、"annos"，
        image_input_instdiff = {
            "caption": image_caption,
            "width": width,
            "height": height,
            "annos": ann,
        }
        #instance_diffusion生成image_data_instdiff的图片
        generate_image(image_input_instdiff, target_domain, images_data['file_name'], model, autoencoder, text_encoder, diffusion, config,clip_model, clip_processor, grounding_tokenizer_input)
        # return image_input_instdiff




if __name__ == '__main__':

    print("hello")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,  default="OUTPUT/cityscapes2daytimeclear", help="root folder for output")
    parser.add_argument("--num_images", type=int, default=1, help="")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--input_json", type=str, default='demos/demo_cat_dog_robin.json', help="json files for instance-level conditions")
    parser.add_argument("--ckpt", type=str, default='pretrained/cityscapes_500_40e_instdiff.pth', help="pretrained checkpoint")
    # pretrained/instancediffusion_sd15.pth
    # code/InstanceDiffusion/OUTPUT/checkpoint-01/tag04/checkpoint_00010010.pth
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--alpha", type=float, default=1, help="the percentage of timesteps using grounding inputs")
    parser.add_argument("--mis", type=float, default=0.36, help="the percentage of timesteps using MIS")
    parser.add_argument("--cascade_strength", type=float, default=0, help="strength of SDXL Refiner.")
    parser.add_argument("--test_config", type=str, default="configs/test_mask.yaml", help="config for model inference.")

    parser.add_argument("--source_json_path", type=str, default="/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_centercrop/voc07_train_centercrop.json", help="config for model inference.")
    parser.add_argument("--source_image_path", type=str, default="/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_centercrop/")
    parser.add_argument("--target_domain", type=str, default="daytimefoggy")

    args = parser.parse_args()
    return_att_masks = False
    ckpt = args.ckpt

    seed = args.seed
    # if args.target_domain == "nightrainy":
    #     seed = 20
    # # if args.target_domain == "daytimefoggy":
    # #     seed = 42
    # if args.target_domain == "daytimefoggy":
    #     seed = 123
    # if args.target_domain == "duskrainy":
    #     seed = 42
    # if args.target_domain == "nightsunny":
    #     seed = 42
    # if args.target_domain == "daytimeclear":
    #     seed = 0


    save_folder_name = f"gc{args.guidance_scale}-seed{seed}-alpha{args.alpha}"

    # get attribute injection input
    image_id_to_annotations = get_image_id_to_annotations('/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_crop/voc07_train_crop.json')
    source_image_dir = args.source_image_path
    source_json_path = args.source_json_path

    model, autoencoder, text_encoder, diffusion, config = load_model_ckpt(ckpt, args, device)
    clip_model, clip_processor = create_clip_pretrain_model()

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    generate_data(source_image_dir, source_json_path, args.target_domain, model, autoencoder, text_encoder, diffusion, config,clip_model, clip_processor, grounding_tokenizer_input)




