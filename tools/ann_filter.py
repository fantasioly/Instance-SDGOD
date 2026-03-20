#1.读取coco标注格式的json文件，读取图片文件夹，根据图片名称，删除掉json中没有的图片
#2.遍历每张图片，遍历该图片的标注信息，提取标注的每个box，计算区域ccmd的值，如果ccmd大于阈值，则删除该box

import json
import os
import numpy as np
from tools.cmmd import distance, embedding
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from skimage.metrics import structural_similarity as ssim

def compute_ssim(imageA, imageB, bboxA, bboxB):
    # Convert images to grayscale
    imageA = cv2.imread(imageA)
    #resize image
    imageB = cv2.imread(imageB)
    source_region = imageA[bboxA[0]:bboxA[0]+bboxA[2], bboxA[1]:bboxA[1]+bboxA[3]]
    virtual_region = imageB[bboxB[0]:bboxB[0]+bboxB[2], bboxB[1]:bboxB[1]+bboxB[3]]
    source_region = cv2.resize(source_region, (bboxB[3], bboxB[2]))
    grayA = cv2.cvtColor(source_region, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(virtual_region, cv2.COLOR_BGR2GRAY)
    # Compute SSIM between two images
    score, _ = ssim(grayA, grayB, full=True)
    return score

def _center_crop_and_resize(im, size):
      w, h = im.size
      l = min(w, h)
      top = (h - l) // 2
      left = (w - l) // 2
      box = (left, top, left + l, top + l)
      im = im.crop(box)
      # Note that the following performs anti-aliasing as well.
      return im.resize((size, size), resample=Image.BICUBIC)

def _resize_bicubic(images, size):
#如果images的维度是3，name在第0轴添加一个维度表示是batch
    if len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images

def process_region(img_path, bbox, reshapeto=336):
    im = Image.open(img_path)
    cropped_im = im.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))

    plt.imshow(cropped_im)
    plt.show()

    if reshapeto > 0:
        im = cropped_im.resize((reshapeto, reshapeto), resample=Image.BICUBIC)
        x= np.asarray(im).astype(np.float32)
        img = _resize_bicubic(x, reshapeto)
        inputs = image_processor(
            images=img,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        if _CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            image_embs = embedding_model._model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs

def process_img(img_path, reshapeto=336):
    im = Image.open(img_path)
    plt.imshow(im)
    plt.show()

    if reshapeto > 0:
        im = im.resize((reshapeto, reshapeto), resample=Image.BICUBIC)
        x= np.asarray(im).astype(np.float32)
        img = _resize_bicubic(x, reshapeto)
        inputs = image_processor(
            images=img,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        if _CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            image_embs = embedding_model._model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs

def compute_cmmd_bbox(ref, eval, ref_box, targt_box):
      embs_ref = process_region(ref, ref_box, reshapeto=336,)
      embs_eval = process_region(eval, targt_box, reshapeto=336)
      embs_ref = embs_ref.detach().numpy()
      embs_eval = embs_eval.detach().numpy()
      val = distance.mmd(embs_ref, embs_eval).numpy()
      return val

def compute_cmmd_img(ref, eval):
    embs_ref = process_img(ref, reshapeto=336)
    embs_eval = process_img(eval, reshapeto=336)
    embs_ref = embs_ref.detach().numpy()
    embs_eval = embs_eval.detach().numpy()
    val = distance.mmd(embs_ref, embs_eval).numpy()
    return val

def filter_coco_annotation_by_images(coco_annotation_file, images_folder, output_annotation_file):
      # 读取COCO标注文件
      with open(coco_annotation_file, 'r') as f:
            coco_data = json.load(f)

      # 获取图片文件夹中的图片文件名列表
      image_files = ['VOC2007/JPEGImages/'.lower() + f.lower() for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

      # 筛选出存在于图片文件夹中的图片的标注信息
      filtered_images = set(img['file_name'] for img in coco_data['images'] if img['file_name'].lower() in image_files)
      #构建image id和filtered_images映射
      image_id_to_name = {img['id']: img['file_name'] if img['file_name'] in filtered_images else None for img in coco_data['images']}

      #图片缩放因子为实际图片的宽度除以标注文件中的宽度，取整
      scale_factor = 512/720

      # 构建新的标注信息列表
      new_annotations = []
      for annotation in tqdm(coco_data['annotations']):
            if annotation['image_id'] in image_id_to_name and image_id_to_name[annotation['image_id']] is not None:
                  #提取源域图像区域和目标域图像区域
                  # if annotation['id']<85:
                  #     continue

                  #cmmd filter
                  # source_img = source_img_crop_path + image_id_to_name[annotation['image_id']]
                  # target_img = target_img_crop_path + image_id_to_name[annotation['image_id']]
                  # ref_box = annotation['bbox']
                  # targt_box = [int(b * scale_factor) for b in annotation['bbox']]
                  # cmmd_value = compute_cmmd_bbox(source_img, target_img, ref_box, targt_box)


                  # if cmmd_value > 2.5:
                  #       continue


                  #更新bbox
                  annotation['bbox'] = [int(b * scale_factor) for b in annotation['bbox']]
                  new_annotations.append(annotation)
      # 更新COCO数据中的annotations部分
      coco_data['annotations'] = new_annotations

      # 移除不存在于图片文件夹中的图片信息
      coco_data['images'] = [img for img in coco_data['images'] if img['file_name'] in filtered_images]
      #更新coco_data['images']中的height和width
      for img in coco_data['images']:
            img['height'] = 512
            img['width'] = 512

      # 保存新的标注文件
      with open(output_annotation_file, 'w') as f:
            json.dump(coco_data, f, indent=4)


# 使用示例
target_img_crop_path = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/daytimeclear/'

source_img_crop_path = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_centercrop/'
coco_annotation_file = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_centercrop/voc07_train_centercrop.json'  # COCO标注文件路径

images_folder0 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/daytimeclear/VOC2007/JPEGImages'  # 图片文件夹路径
output_annotation_file0 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/daytimeclear/voc07_train_centercrop.json'  # 输出标注文件保存路径


images_folder1 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/daytimefoggy/VOC2007/JPEGImages'  # 图片文件夹路径
output_annotation_file1 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/daytimefoggy/voc07_train_centercrop.json'  # 输出标注文件保存路径

images_folder2 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/duskrainy/VOC2007/JPEGImages'  # 图片文件夹路径
output_annotation_file2 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/duskrainy/voc07_train_centercrop.json'  # 输出标注文件保存路径

images_folder3 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/nightrainy/VOC2007/JPEGImages'  # 图片文件夹路径
output_annotation_file3 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/nightrainy/voc07_train_centercrop.json'  # 输出标注文件保存路径

images_folder4 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/nightsunny/VOC2007/JPEGImages'  # 图片文件夹路径
output_annotation_file4 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytimeclear_instdiff/nightsunny/voc07_train_centercrop.json'  # 输出标注文件保存路径

# images_folder4 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/rare_classes_generation/VOC2007/JPEGImages'  # 图片文件夹路径
# output_annotation_file4 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/rare_classes_generation//train.json'



if __name__ == '__main__':
    embedding_model = embedding.ClipEmbeddingModel()
    _CLIP_MODEL_NAME = "/cpfs/user/haoli84/environment/clip-vit-large-patch14-336/"
    _CUDA_AVAILABLE = torch.cuda.is_available()
    image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)
    filter_coco_annotation_by_images(coco_annotation_file, images_folder0, output_annotation_file0)
    filter_coco_annotation_by_images(coco_annotation_file, images_folder1, output_annotation_file1)
    filter_coco_annotation_by_images(coco_annotation_file, images_folder2, output_annotation_file2)
    filter_coco_annotation_by_images(coco_annotation_file, images_folder3, output_annotation_file3)
    filter_coco_annotation_by_images(coco_annotation_file, images_folder4, output_annotation_file4)