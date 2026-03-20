import json
from collections import defaultdict, Counter


def get_least_common_categories_indices(json_file, n=4):
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize category count dictionary
    category_count = Counter()
    # Initialize dictionary mapping categories to image indices
    category_indices = defaultdict(set)

    # Extract category ID to name mapping
    categories = {category['id']: category['name'] for category in data['categories']}

    # Create image ID to index mapping
    image_id_to_index = {image['id']: idx for idx, image in enumerate(data['images'])}

    # Iterate through all annotations, count each category and record image indices
    for annotation in data['annotations']:
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        category_count[category_id] += 1
        category_indices[category_id].add(image_id_to_index[image_id])

    # Find the n least common categories
    least_common_categories = category_count.most_common()[:-n - 1:-1]

    # Get union of image indices for the n least common categories
    index_union = set()
    for category_id, count in least_common_categories:
        index_union.update(category_indices[category_id])
    index_union = sorted(index_union)

    # Initialize annotations for each image index
    image_annotations = defaultdict(list)
    for annotation in data['annotations']:
        image_annotations[image_id_to_index[annotation['image_id']]].append(annotation)

    # Count combined category annotations for these images
    combined_category_count = Counter()
    for index in index_union:
        for annotation in image_annotations[index]:
            combined_category_count[annotation['category_id']] += 1

    # Print results
    print(f"Union of Image Indices: {list(index_union)}")
    print("Combined Category Counts:")
    for category_id, count in combined_category_count.items():
        print(f"Category: {categories[category_id]}, Count: {count}")


def count_categories(json_file):
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize category count dictionary
    category_count = defaultdict(int)

    # Extract category ID to name mapping
    categories = {category['id']: category['name'] for category in data['categories']}

    # Iterate through all annotations and count each category
    for annotation in data['annotations']:
        category_id = annotation['category_id']
        category_count[category_id] += 1

    # Print count for each category
    for category_id, count in category_count.items():
        print(f"Category: {categories[category_id]}, Count: {count}")

if __name__ == '__main__':
    # Usage example
    json_file0 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear/voc07_train.json'
    json_file1 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear_crop/voc07_train_crop.json'
    json_file2 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/generated_v3/daytimeclear/voc07_train_crop_filter.json'
    json_file3 = '/cpfs/user/haoli84/code/Datasets/Adverse-Weather/generated_v3/daytimeclear/voc07_train_crop_filter_cmmd.json'
    # Replace with your COCO JSON annotation file path
    # count_categories(json_file0)
    # print("===========================================================================================================")
    # count_categories(json_file1)
    # print("===========================================================================================================")
    # count_categories(json_file2)
    # print("===========================================================================================================")
    count_categories(json_file3)
    print("===========================================================================================================")
    get_least_common_categories_indices(json_file3)
