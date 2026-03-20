import json
from collections import defaultdict, Counter


def get_least_common_categories_indices(json_file, n=4):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 初始化类别计数字典
    category_count = Counter()
    # 初始化类别对应图片索引的字典
    category_indices = defaultdict(set)

    # 提取类别ID与名称的映射
    categories = {category['id']: category['name'] for category in data['categories']}

    # 创建图片ID到索引的映射
    image_id_to_index = {image['id']: idx for idx, image in enumerate(data['images'])}

    # 遍历所有注释，统计每个类别的数量并记录图片索引
    for annotation in data['annotations']:
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        category_count[category_id] += 1
        category_indices[category_id].add(image_id_to_index[image_id])

    # 找出数量最少的6个类别
    least_common_categories = category_count.most_common()[:-n - 1:-1]

    # 获取最少6个类别的图片索引列表并取并集
    index_union = set()
    for category_id, count in least_common_categories:
        index_union.update(category_indices[category_id])
    index_union = sorted(index_union)

    # 初始化图片索引对应的所有注释
    image_annotations = defaultdict(list)
    for annotation in data['annotations']:
        image_annotations[image_id_to_index[annotation['image_id']]].append(annotation)

    # 统计这些图片所包含每种类别的总标注数量
    combined_category_count = Counter()
    for index in index_union:
        for annotation in image_annotations[index]:
            combined_category_count[annotation['category_id']] += 1

    # 打印结果
    print(f"Union of Image Indices: {list(index_union)}")
    print("Combined Category Counts:")
    for category_id, count in combined_category_count.items():
        print(f"Category: {categories[category_id]}, Count: {count}")


def count_categories(json_file):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 初始化类别计数字典
    category_count = defaultdict(int)

    # 提取类别ID与名称的映射
    categories = {category['id']: category['name'] for category in data['categories']}

    # 遍历所有注释，统计每个类别的数量
    for annotation in data['annotations']:
        category_id = annotation['category_id']
        category_count[category_id] += 1

    # 打印每个类别的数量
    for category_id, count in category_count.items():
        print(f"Category: {categories[category_id]}, Count: {count}")

if __name__ == '__main__':
    # 使用示例
    json_file0 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear/voc07_train.json'
    json_file1 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_crop/voc07_train_crop.json'
    json_file2 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/generated_v3/daytimeclear/voc07_train_crop_filter.json'
    json_file3 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/generated_v3/daytimeclear/voc07_train_crop_filter_cmmd.json'
    # 替换为你的COCO JSON标注文件路径
    # count_categories(json_file0)
    # print("===========================================================================================================")
    # count_categories(json_file1)
    # print("===========================================================================================================")
    # count_categories(json_file2)
    # print("===========================================================================================================")
    count_categories(json_file3)
    print("===========================================================================================================")
    get_least_common_categories_indices(json_file3)
