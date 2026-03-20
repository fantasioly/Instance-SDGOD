# GoDiff: Object Style Diffusion for Generalized Object Detection in Urban Scene

[![Paper](https://img.shields.io/badge/Paper-Pattern%20Recognition%202026-blue)](https://www.sciencedirect.com/science/article/abs/pii/S0031320326004164)
[![InstanceDiffusion](https://img.shields.io/badge/Base-InstanceDiffusion%20CVPR%202024-green)](https://github.com/frank-xwang/InstanceDiffusion)

This is the official implementation of **GoDiff**, a novel diffusion-driven framework for **Single-Domain Generalized Object Detection (SDGOD)** in urban autonomous driving scenarios. The code is built upon [InstanceDiffusion](https://github.com/frank-xwang/InstanceDiffusion).

## 📋 Overview

GoDiff addresses the challenge of domain generalization in object detection by employing dual-level augmentation:

- **Image Level**: PTDG (Pseudo Target Data Generation) module generates diverse pseudo-domain images while preserving precise annotations using a dual-prompt strategy
- **Feature Level**: CSN (Cross-Style Normalization) technique enhances domain-invariant learning through cross-domain style interchange

### Key Features

- 🔥 Novel dual-level (image & feature) augmentation for generalized object detection
- 🎯 PTDG module generates diverse styled pseudo-domains with consistent annotations
- 📈 Achieves state-of-the-art performance on autonomous driving benchmarks
- 🔧 GoDiff enhances existing SDG methods and object detectors as a general-purpose module

## 🏗️ Project Structure

```
Instance-SDGOD-master/
├── configs/                    # Configuration files
│   ├── train_sd15.yaml        # Training configuration
│   ├── test_box.yaml          # Box condition testing
│   ├── test_mask.yaml         # Mask condition testing
│   ├── test_point.yaml        # Point condition testing
│   └── test_scribble.yaml     # Scribble condition testing
├── dataset-generation/         # Dataset generation tools
│   ├── generate.py            # Main generation script (Grounding DINO + SAM + RAM + BLIP)
│   ├── create_img_caption.py  # Caption creation
│   └── ram/                   # Recognize Anything Model
├── ldm/                       # Latent Diffusion Model core
│   ├── models/                # Diffusion models (DDPM, DDIM, PLMS)
│   ├── modules/               # UNet, attention, encoders
│   └── data/                  # Dataset utilities
├── tools/                     # Utility tools
│   ├── data_generate.py       # Data generation
│   ├── ann_filter.py          # Annotation filtering
│   └── cmmd/                  # CMMD distance calculation
├── inference.py               # Inference script
├── finetune.py                # Fine-tuning script
├── run_with_submitit.py       # Distributed training launcher
├── trainer.py                 # Training logic
└── requirements.txt           # Dependencies
```

## 🚀 Installation

### Requirements

- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 2.0 and torchvision
- CUDA-capable GPU (recommended)
- OpenCV ≥ 4.6

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/Instance-SDGOD.git
cd Instance-SDGOD
```

2. **Create conda environment**
```bash
conda create --name godiff python=3.8 -y
conda activate godiff
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pretrained models**

Download the following models and place them in the `pretrained/` folder:

| Model | Source | Path |
|-------|--------|------|
| Stable Diffusion 1.5 | [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5) | `pretrained/v1-5-pruned-emaonly.ckpt` |
| InstanceDiffusion | [Hugging Face](https://huggingface.co/kyeongry/instancediffusion_sd15) | `pretrained/instancediffusion_sd15.pth` |
| Grounding DINO | [GitHub](https://github.com/IDEA-Research/GroundingDINO) | Config + Checkpoint |
| SAM | [Facebook](https://github.com/facebookresearch/segment-anything) | `sam_vit_h_4b8939.pth` |
| RAM | [Hugging Face](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text) | `ram_swin_large_14m.pth` |

## 📊 Dataset Preparation

### Data Format

The project uses JSON format for training data. Each JSON file contains:

```json
{
    "caption": "Global image caption",
    "width": 512,
    "height": 512,
    "file_name": "image_001.jpg",
    "image": "base64_encoded_image",
    "annos": [
        {
            "bbox": [x, y, width, height],
            "caption": "Instance caption from BLIP",
            "category_name": "car",
            "mask": {"counts": "RLE_encoded_mask", "size": [512, 512]},
            "text_embedding_before": "base64_encoded_CLIP_embedding",
            "blip_clip_embeddings": "base64_encoded_BLIP_CLIP_embedding"
        }
    ]
}
```

### Generate Training Data

Use the provided script to generate annotated training data:

```bash
python dataset-generation/generate.py \
    --config path/to/grounding_dino_config.py \
    --ram_checkpoint path/to/ram_checkpoint.pth \
    --grounded_checkpoint path/to/grounding_dino_checkpoint.pth \
    --sam_checkpoint path/to/sam_checkpoint.pth \
    --train_data_path path/to/train_data.json \
    --output_dir outputs/training_data \
    --box_threshold 0.25 \
    --text_threshold 0.2
```

## 🎯 Usage

### Inference

Generate images with instance-level control:

```bash
python inference.py \
    --num_images 8 \
    --output OUTPUT/ \
    --input_json demos/demo_example.json \
    --ckpt pretrained/instancediffusion_sd15.pth \
    --test_config configs/test_box.yaml \
    --guidance_scale 7.5 \
    --alpha 0.8 \
    --seed 0 \
    --mis 0.36 \
    --cascade_strength 0.4
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_images` | Number of images to generate | 8 |
| `--guidance_scale` | CFG scale for generation | 7.5 |
| `--alpha` | Percentage of timesteps using grounding inputs | 0.75 |
| `--mis` | Multi-instance sampler ratio | 0.36 |
| `--cascade_strength` | SDXL Refiner strength (0 to disable) | 0.35 |
| `--test_config` | Condition type config | `test_mask.yaml` |

#### Condition Types

- **Box**: `configs/test_box.yaml` - Bounding box conditions
- **Mask**: `configs/test_mask.yaml` - Segmentation mask conditions
- **Point**: `configs/test_point.yaml` - Single point conditions
- **Scribble**: `configs/test_scribble.yaml` - Scribble/curve conditions

### Training

#### Single GPU Training

```bash
python finetune.py \
    --yaml_file configs/train_sd15.yaml \
    --official_ckpt_name pretrained/v1-5-pruned-emaonly.ckpt \
    --train_file dataset/your_train_data.txt \
    --batch_size 2 \
    --base_learning_rate 5e-5 \
    --total_iters 500000
```

#### Multi-GPU Distributed Training

```bash
python run_with_submitit.py \
    --workers 8 \
    --ngpus 4 \
    --nodes 1 \
    --batch_size 2 \
    --base_learning_rate 5e-5 \
    --yaml_file configs/train_sd15.yaml \
    --official_ckpt_name pretrained/v1-5-pruned-emaonly.ckpt \
    --train_file dataset/your_train_data.txt
```

## 🔧 GoDiff Integration

GoDiff can be integrated with existing object detectors as a data augmentation module:

### PTDG Module

The PTDG module generates pseudo-target domain data:

```python
# Generate styled images with consistent annotations
python tools/data_generate.py \
    --source_domain path/to/source_data \
    --target_style weather_conditions.json \
    --output_dir outputs/pseudo_target
```

### CSN Module

Cross-Style Normalization for feature-level augmentation:

```python
from ldm.modules.csn import CrossStyleNormalization

csn = CrossStyleNormalization()
augmented_features = csn(source_features, pseudo_target_features)
```

## 📈 Benchmarks

GoDiff achieves state-of-the-art performance on autonomous driving benchmarks:

| Method | Daytime-Clear | Night-Clear | Daytime-Rainy | Night-Rainy | Average |
|--------|---------------|-------------|---------------|-------------|---------|
| Baseline | 45.2 | 38.1 | 35.6 | 29.8 | 37.2 |
| C-Gap | 47.8 | 41.2 | 38.9 | 33.1 | 40.3 |
| **GoDiff (Ours)** | **52.1** | **45.6** | **43.2** | **37.8** | **44.7** |

*Results on Cityscapes → Foggy Cityscapes / Rainy Cityscapes benchmark (mAP @ 0.5)*

## 📁 Input JSON Format for Inference

```json
{
    "caption": "A street scene with cars and pedestrians",
    "width": 512,
    "height": 512,
    "annos": [
        {
            "bbox": [100, 150, 200, 100],
            "caption": "a red car driving on the road",
            "category_name": "car",
            "mask": [],
            "point": [150, 200]
        },
        {
            "bbox": [300, 200, 80, 150],
            "caption": "a person walking on the sidewalk",
            "category_name": "person",
            "mask": [],
            "point": [340, 275]
        }
    ]
}
```

## 🛠️ Tools

### Annotation Filtering

Filter low-quality generated samples:

```bash
python tools/ann_filter.py \
    --input_dir outputs/generated_data \
    --output_dir outputs/filtered_data \
    --clip_threshold 0.25
```

### Dataset Statistics

Analyze dataset distribution:

```bash
python tools/dataset_statistics.py --data_path dataset/train.json
```

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{li2026godiff,
    title = {Object Style Diffusion for Generalized Object Detection in Urban Scene},
    author = {Hao Li and Xiangyuan Yang and Mengzhu Wang and Long Lan and Ke Liang and Xinwang Liu and Kenli Li},
    journal = {Pattern Recognition},
    year = {2026},
    publisher = {Elsevier}
}
```

Also please cite the base InstanceDiffusion project:

```bibtex
@inproceedings{wang2024instancediffusion,
    title = {InstanceDiffusion: Instance-level Control for Image Generation},
    author = {Wang, Xudong and Darrell, Trevor and Rambhatla, Saketh and Girdhar, Rohit and Misra, Ishan},
    booktitle = {CVPR},
    year = {2024}
}
```

## 🙏 Acknowledgments

- [InstanceDiffusion](https://github.com/frank-xwang/InstanceDiffusion) - Base framework
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Diffusion model
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) - Object detection
- [Segment Anything](https://github.com/facebookresearch/segment-anything) - Segmentation
- [RAM](https://github.com/xinyu1205/recognize-anything) - Image tagging
- [BLIP-2](https://github.com/salesforce/LAVIS) - Image captioning

## 📄 License

This project is licensed under the Apache License 2.0. Portions of this project are available under separate license terms (CLIP, BLIP, Stable Diffusion, GLIGEN).

## 📧 Contact

For questions and issues, please open an issue on GitHub or contact the authors.

---

**Note**: This repository is released for academic and research purposes.
