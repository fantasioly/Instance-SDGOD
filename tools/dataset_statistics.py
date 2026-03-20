import xml.etree.ElementTree as ET
from collections import defaultdict
import os


def parse_xml(xml_file):
    # 用于存储各属性和对象统计的字典
    stats = {
        'weather': defaultdict(int),
        'scene': defaultdict(int),
        'timeofday': defaultdict(int),
        'object_count': 0,
        'object_types': defaultdict(int),
        'occluded': defaultdict(int),
        'truncated': defaultdict(int),
        'trafficLightColor': defaultdict(int)
    }

    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 统计attributes中的weather, scene, timeofday
    attributes = root.find('attributes')
    if attributes is not None:
        stats['weather'][attributes.find('weather').text] += 1
        stats['scene'][attributes.find('scene').text] += 1
        stats['timeofday'][attributes.find('timeofday').text] += 1

    # 统计objects
    for obj in root.findall('object'):
        stats['object_count'] += 1
        obj_type = obj.find('name').text
        stats['object_types'][obj_type] += 1

        obj_attributes = obj.find('attributes')
        stats['occluded'][obj_attributes.find('occluded').text] += 1
        stats['truncated'][obj_attributes.find('truncated').text] += 1
        stats['trafficLightColor'][obj_attributes.find('trafficLightColor').text] += 1

    return stats
def aggregate_daytimeclear_stats(txt_file, xml_folder):
    # 总体统计
    total_stats = {
        'weather': defaultdict(int),
        'scene': defaultdict(int),
        'timeofday': defaultdict(int),
        'object_count': 0,
        'object_types': defaultdict(int),
        'occluded': defaultdict(int),
        'truncated': defaultdict(int),
        'trafficLightColor': defaultdict(int)
    }

    # 读取训练集文件名
    with open(txt_file, 'r') as file:
        for line in file:
            xml_file = os.path.join(xml_folder, line.strip() + '.xml')
            stats = parse_xml(xml_file)

            # 累加统计结果
            for key in total_stats:
                if key in ['weather', 'scene', 'timeofday', 'object_types', 'occluded', 'truncated',
                           'trafficLightColor']:
                    for subkey, count in stats[key].items():
                        total_stats[key][subkey] += count
                else:
                    total_stats[key] += stats[key]



    return total_stats


def parse_daytimefoggy_xml(xml_file):
    # 用于存储各属性和对象统计的字典
    stats = {
        'object_count': 0,
        'object_types': defaultdict(int),
        'pose': defaultdict(int),
        'truncated': defaultdict(int),
        'difficult': defaultdict(int),
    }

    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 统计objects
    for obj in root.findall('object'):
        stats['object_count'] += 1
        obj_type = obj.find('name').text
        stats['object_types'][obj_type] += 1
        stats['pose'][obj.find('pose').text] += 1
        stats['truncated'][obj.find('truncated').text] += 1
        stats['difficult'][obj.find('difficult').text] += 1
    return stats
def aggregate_daytimefoggy_stats(txt_file, xml_folder):
    # 总体统计
    total_stats = {
        'object_count': 0,
        'object_types': defaultdict(int),
        'pose': defaultdict(int),
        'truncated': defaultdict(int),
        'difficult': defaultdict(int),
    }

    # 读取训练集文件名
    with open(txt_file, 'r') as file:
        for line in file:
            xml_file = os.path.join(xml_folder, line.strip() + '.xml')
            stats = parse_daytimefoggy_xml(xml_file)

            # 累加统计结果
            for key in total_stats:
                if key in ['object_types', 'pose', 'truncated', 'difficult']:
                    for subkey, count in stats[key].items():
                        total_stats[key][subkey] += count
                else:
                    total_stats[key] += stats[key]

    return total_stats

def parse_duskrainy_xml(xml_file):
    # 用于存储各属性和对象统计的字典
    stats = {
        'weather': defaultdict(int),
        'scene': defaultdict(int),
        'timeofday': defaultdict(int),
        'object_count': 0,
        'object_types': defaultdict(int),
        'difficult': defaultdict(int),
        'truncated': defaultdict(int),
        'pose': defaultdict(int)
    }

    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 统计attributes中的weather, scene, timeofday
    attributes = root.find('attributes')
    if attributes is not None:
        stats['weather'][attributes.find('weather').text] += 1
        stats['scene'][attributes.find('scene').text] += 1
        stats['timeofday'][attributes.find('timeofday').text] += 1

    # 统计objects
    for obj in root.findall('object'):
        stats['object_count'] += 1
        obj_type = obj.find('name').text
        stats['object_types'][obj_type] += 1
        stats['difficult'][obj.find('difficult').text] += 1
        stats['truncated'][obj.find('truncated').text] += 1
        stats['pose'][obj.find('pose').text] += 1

    return stats
def aggregate_duskrainy_stats(txt_file, xml_folder):
    # 总体统计
    total_stats = {
        'weather': defaultdict(int),
        'scene': defaultdict(int),
        'timeofday': defaultdict(int),
        'object_count': 0,
        'object_types': defaultdict(int),
        'difficult': defaultdict(int),
        'truncated': defaultdict(int),
        'pose': defaultdict(int)
    }

    # 读取训练集文件名
    with open(txt_file, 'r') as file:
        for line in file:
            xml_file = os.path.join(xml_folder, line.strip() + '.xml')
            stats = parse_duskrainy_xml(xml_file)

            # 累加统计结果
            for key in total_stats:
                if key in ['weather', 'scene', 'timeofday', 'object_types', 'difficult', 'truncated',
                           'pose']:
                    for subkey, count in stats[key].items():
                        total_stats[key][subkey] += count
                else:
                    total_stats[key] += stats[key]



    return total_stats

def parse_nightrainy_xml(xml_file):
    # 用于存储各属性和对象统计的字典
    stats = {
        'weather': defaultdict(int),
        'scene': defaultdict(int),
        'timeofday': defaultdict(int),
        'object_count': 0,
        'object_types': defaultdict(int),
        'difficult': defaultdict(int),
        'truncated': defaultdict(int),
        'pose': defaultdict(int)
    }

    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 统计attributes中的weather, scene, timeofday
    attributes = root.find('attributes')
    if attributes is not None:
        stats['weather'][attributes.find('weather').text] += 1
        stats['scene'][attributes.find('scene').text] += 1
        stats['timeofday'][attributes.find('timeofday').text] += 1

    # 统计objects
    for obj in root.findall('object'):
        stats['object_count'] += 1
        obj_type = obj.find('name').text
        stats['object_types'][obj_type] += 1
        stats['difficult'][obj.find('difficult').text] += 1
        stats['truncated'][obj.find('truncated').text] += 1
        stats['pose'][obj.find('pose').text] += 1

    return stats
def aggregate_nightrainy_stats(txt_file, xml_folder):
    # 总体统计
    total_stats = {
        'weather': defaultdict(int),
        'scene': defaultdict(int),
        'timeofday': defaultdict(int),
        'object_count': 0,
        'object_types': defaultdict(int),
        'difficult': defaultdict(int),
        'truncated': defaultdict(int),
        'pose': defaultdict(int)
    }

    # 读取训练集文件名
    with open(txt_file, 'r') as file:
        for line in file:
            xml_file = os.path.join(xml_folder, line.strip() + '.xml')
            stats = parse_nightrainy_xml(xml_file)

            # 累加统计结果
            for key in total_stats:
                if key in ['weather', 'scene', 'timeofday', 'object_types', 'difficult', 'truncated',
                           'pose']:
                    for subkey, count in stats[key].items():
                        total_stats[key][subkey] += count
                else:
                    total_stats[key] += stats[key]



    return total_stats


# 示例使用：
# results = aggregate_daytimeclear_stats('/home/haoli84/code/Datasets/Adverse-Weather/daytime_clear/VOC2007/ImageSets/Main/train.txt',
#                 '/home/haoli84/code/Datasets/Adverse-Weather/daytime_clear/VOC2007/Annotations')

# results = aggregate_daytimefoggy_stats('/home/haoli84/code/Datasets/Adverse-Weather/daytime_foggy/VOC2007/ImageSets/Main/test.txt',
#                 '/home/haoli84/code/Datasets/Adverse-Weather/daytime_foggy/VOC2007/Annotations')

# results = aggregate_duskrainy_stats('/home/haoli84/code/Datasets/Adverse-Weather/dusk_rainy/VOC2007/ImageSets/Main/train.txt',
#                 '/home/haoli84/code/Datasets/Adverse-Weather/dusk_rainy/VOC2007/Annotations')

results = aggregate_nightrainy_stats('/home/haoli84/code/Datasets/Adverse-Weather/night_sunny/VOC2007/ImageSets/Main/test.txt',
                '/home/haoli84/code/Datasets/Adverse-Weather/night_sunny/VOC2007/Annotations')


for key, value in results.items():
    if isinstance(value, defaultdict):
        print(f"{key}:")
        for subkey, count in value.items():
            print(f"  {subkey}: {count}")
    else:
        print(f"{key}: {value}")