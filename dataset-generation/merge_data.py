import json


def merge_json_files(file1_path, file2_path, output_file_path):
    # Read first JSON file
    with open(file1_path, 'r') as file1:
        data1 = json.load(file1)

    # Read second JSON file
    with open(file2_path, 'r') as file2:
        data2 = json.load(file2)

    # Merge two data structures (assuming concatenation of two lists)
    merged_data = data1 + data2

    # Write to new JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=4)


# Use function
file1_path = '/cpfs/user/haoli84/code/InstanceDiffusion/cityscapes_image_paths_with_caption.json'
file2_path = '/cpfs/user/haoli84/code/InstanceDiffusion/daytimeclear_image_paths_with_caption.json'
output_file_path = 'merged_train_data.json'

merge_json_files(file1_path, file2_path, output_file_path)