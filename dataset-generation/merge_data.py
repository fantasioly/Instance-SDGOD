import json


def merge_json_files(file1_path, file2_path, output_file_path):
    # 读取第一个JSON文件
    with open(file1_path, 'r') as file1:
        data1 = json.load(file1)

    # 读取第二个JSON文件
    with open(file2_path, 'r') as file2:
        data2 = json.load(file2)

    # 合并两个数据结构（这里假设是将两个列表相加）
    merged_data = data1 + data2

    # 写入到新的JSON文件
    with open(output_file_path, 'w') as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=4)


# 使用函数
file1_path = '/cpfs/user/haoli84/code/InstanceDiffusion/cityscapes_image_paths_with_caption.json'
file2_path = '/cpfs/user/haoli84/code/InstanceDiffusion/daytimeclear_image_paths_with_caption.json'
output_file_path = 'merged_train_data.json'

merge_json_files(file1_path, file2_path, output_file_path)