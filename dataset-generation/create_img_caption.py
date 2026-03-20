import os
import json
import argparse
def get_image_paths_and_save_json(directory, output_json_file):
    # 初始化图片路径字典列表
    image_dicts = []

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否为图片
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                # 构建完整的图片路径
                image_path = os.path.join(root, file)
                # 创建一个字典，包含图片路径和空的caption
                image_dict = {"image": image_path, "caption": ""}
                # 添加到列表
                image_dicts.append(image_dict)

    # 保存到JSON文件
    with open(output_json_file, 'w') as json_file:
        json.dump(image_dicts, json_file, ensure_ascii=False, indent=4)

# 示例用法
# directory = '/cpfs/user/haoli84/code/Datasets/cityscapes/leftImg8bit'  # 替换为你的目录路径
# output_json_file = 'image_paths_with_caption.json'  # 输出的JSON文件名
# get_image_paths_and_save_json(directory, output_json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get image paths and save to JSON.")
    parser.add_argument("--directory", type=str, default='/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear/VOC2007',help="Path to the directory containing images.")
    parser.add_argument("--output_json_file", type=str, default='img_path_caption/daytimeclear_image_paths_with_caption.json', help="Output JSON file name.")

    args = parser.parse_args()

    directory = args.directory
    output_json_file = args.output_json_file

    get_image_paths_and_save_json(directory, output_json_file)