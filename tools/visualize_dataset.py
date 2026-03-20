import os
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# Path to COCO annotations JSON file and images directory
dataset_dir = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_crop'
annotation_file = os.path.join(dataset_dir, 'voc07_train_crop.json')
image_directory = dataset_dir

# Initialize COCO api for instance annotations
coco = COCO(annotation_file)

# Get all image ids in the dataset
image_ids = coco.getImgIds()

# Directory to save visualized images
save_directory = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_crop/visualize'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for img_id in image_ids:
    # Load the image
    img_info = coco.loadImgs(img_id)[0]
    image_path = os.path.join(image_directory, img_info['file_name'])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load annotations
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Plot each annotation
    plt.imshow(image)
    plt.axis('off')
    ax = plt.gca()

    for ann in anns:
        bbox = ann['bbox']
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Save the visualized image
    plt.savefig(os.path.join(save_directory, img_info['file_name']), bbox_inches='tight', pad_inches=0)
    plt.close()

print("Visualization complete. Images saved to", save_directory)
