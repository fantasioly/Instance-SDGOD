import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

def read_annotation(annotation_path):
    with open(annotation_path, 'r') as f:
        return json.load(f)


def update_annotation(ann_dict, img_id, crop):
    if img_id not in ann_dict:
        print(f"Warning: img_id {img_id} not found in ann_dict, creating empty list.")
        return ann_dict

    img_anns = ann_dict[img_id]
    crop_x, crop_y, crop_w, crop_h = crop

    for ann in img_anns:
        x, y, w, h = ann['bbox']

        intersection_x = max(crop_x, x)
        intersection_y = max(crop_y, y)
        intersection_width = min(crop_x + crop_w, x + w) - intersection_x
        intersection_height = min(crop_y + crop_h, y + h) - intersection_y
        intersection_area = intersection_width * intersection_height

        if intersection_area <= 0:
            ann['iscrowd'] = 1
            ann['bbox'] = [0, 0, 0, 0]
        else:
            ann['bbox'] = [intersection_x - crop_x, intersection_y - crop_y,
                           intersection_width, intersection_height]

    return ann_dict


def process_image(image_path, img_anns, img_id, crop):
    img = Image.open(image_path)
    cropped_img = img.crop(crop)
    img_anns = update_annotation(img_anns, img_id, crop)
    return cropped_img, img_anns


def process_img_annotations(imgage, dataset_dir, img_savepath, img_anns):
    img_path = os.path.join(dataset_dir, imgage['file_name'])

    width, height = imgage['width'], imgage['height']
    min_side = min(width, height)
    # left = random.randint(0, width - min_side)
    # top = random.randint(0, height - min_side) if width == min_side else 0
    # right = left + min_side
    # bottom = top + min_side
    # Calculate center crop start position
    left = (width - min_side) // 2
    top = (height - min_side) // 2 if width != min_side else 0
    right = left + min_side
    bottom = top + min_side
    crop = (left, top, right, bottom)

    img, img_anns = process_image(img_path, img_anns, imgage['id'], crop)
    img.save(os.path.join(img_savepath, imgage['file_name']))

    imgage['height'] = min_side
    imgage['width'] = min_side
    return img_anns


def process_dataset(dataset_dir, annotation_path, img_savepath, annotation_savepath):
    dataset = read_annotation(annotation_path)
    images = dataset['images']
    annotations = dataset['annotations']

    ann_dict = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in ann_dict:
            ann_dict[img_id] = []
        ann_dict[img_id].append(ann)

    os.makedirs(img_savepath, exist_ok=True)

    for imgage in tqdm(images, desc='Processing'):
        annotations = process_img_annotations(imgage, dataset_dir, img_savepath, ann_dict)

    crop_ann_dict = []
    id = 0
    for annotation in annotations:
        for ann in annotations[annotation]:
            if ann['bbox'] == [0, 0, 0, 0]:
                continue
            else:
                ann['id'] = id
                crop_ann_dict.append(ann)
                id = id + 1
    dataset['annotations'] = crop_ann_dict

    return dataset


if __name__ == '__main__':

    dataset_dir = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear'
    annotation_path = os.path.join(dataset_dir, 'voc07_train.json')
    img_savepath = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_centercrop'
    annotation_savepath = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_centercrop/voc07_train_centercrop.json'
    dataset = process_dataset(dataset_dir, annotation_path, img_savepath, annotation_savepath)

    with open(annotation_savepath, 'w') as f:
        json.dump(dataset, f)

    print("Image and annotation processing completed.")
