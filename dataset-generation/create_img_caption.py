import os
import json
import argparse
def get_image_paths_and_save_json(directory, output_json_file):
    # Initialize list of image path dictionaries
    image_dicts = []

    # Traverse directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if file is an image
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                # Build full image path
                image_path = os.path.join(root, file)
                # Create a dictionary containing image path and empty caption
                image_dict = {"image": image_path, "caption": ""}
                # Add to list
                image_dicts.append(image_dict)

    # Save to JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(image_dicts, json_file, ensure_ascii=False, indent=4)

# Example usage
# directory = '/cpfs/user/haoli84/code/Datasets/cityscapes/leftImg8bit'  # Replace with your directory path
# output_json_file = 'image_paths_with_caption.json'  # Output JSON file name
# get_image_paths_and_save_json(directory, output_json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get image paths and save to JSON.")
    parser.add_argument("--directory", type=str, default='/cpfs/user/haoli84/code/Datasets/Adverse-Weather/daytime_clear/VOC2007',help="Path to the directory containing images.")
    parser.add_argument("--output_json_file", type=str, default='img_path_caption/daytimeclear_image_paths_with_caption.json', help="Output JSON file name.")

    args = parser.parse_args()

    directory = args.directory
    output_json_file = args.output_json_file

    get_image_paths_and_save_json(directory, output_json_file)