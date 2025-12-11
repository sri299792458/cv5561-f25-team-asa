# Class-Aware Seam Carving with Multi-Operator Retargeting

A modern content-aware image retargeting system that combines **DINOv3 composition classification**, **Depth Anything 3**, **RGB-VST saliency detection**, and **multi-operator retargeting** (crop + scale + seam carving) for intelligent image resizing on the **RetargetMe benchmark**.

---

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/sri299792458/cv5561-f25-team-asa.git
cd cv5561-f25-team-asa

# Create environment (recommended)
conda create -n seam2025 python=3.10 -y
conda activate seam2025
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Depth Anything 3

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3
cd depth-anything-3
pip install -e . --no-deps
cd ..
```

### 4. Set Hugging Face Token (Optional but Recommended)

```bash
export HUGGING_FACE_HUB_TOKEN="your_hf_token_here"
```

### 5. Download RGB-VST Model

```bash
gdown --id 1EksV_DbHt_NGB54C1bR9q8WF-aZHEoGY -O RGB_VST.zip
unzip RGB_VST.zip
```

Expected structure after unzipping:
```
RGB_VST/
├── Models/
├── checkpoint/RGB_VST.pth
└── pretrained_model/80.7_T2T_ViT_t_14.pth.tar
```

### 6. Download RetargetMe Dataset

```bash
mkdir -p data/retargetme
cd data/retargetme
wget https://people.csail.mit.edu/mrub/retargetme/download/images-20100824.zip
unzip images-20100824.zip
cd ../..
```

### 7. Run the Pipeline

```bash
jupyter lab
# Open notebooks/retargetme.ipynb and run all cells
```

---

## Project Structure

```
cv5561-f25-team-asa/
├── notebooks/
│   ├── retargetme.ipynb              # Main pipeline
│   └── train_dinov3_picd.ipynb       # Classifier training
├── checkpoints/
│   └── dinov3/
│       └── best_dinov3_classifier_reduced.pth  # Pre-trained classifier
├── requirements.txt
├── RGB_VST/                           # Download separately
├── depth-anything-3/                  # Git clone (step 3)
├── data/retargetme/images/           # Download (step 6)
└── results/retargetme/               # Output directory
```

---

## Core Components

### 1. DINOv3 Composition Classifier
- **Model**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- **Classes**: 10 composition types (RuleOfThirds, Centered, Diagonal, Horizontal, Vertical, Triangle, Curves, RadialPerspective, DensePattern, Scatter)
- **Checkpoint**: Included in `checkpoints/dinov3/`

### 2. Depth Anything 3 (DA3)
- **Model**: `depth-anything/DA3-LARGE`
- **Output**: Depth map + confidence map
- **Purpose**: Preserve 3D structure during retargeting

### 3. RGB-VST Saliency
- **Model**: T2T-ViT-t-14 transformer
- **Output**: Salient region detection
- **Purpose**: Identify important content areas

### 4. Multi-Operator Retargeting
Optimally combines three operations:
- **Crop**: Remove low-energy borders (up to 10%)
- **Scale**: Global resize (down to 85%)
- **Seam Carving**: Content-aware pixel removal

---

## Key Features

### Class-Aware Energy Maps
Each composition class has a custom energy recipe:
- **Rule of Thirds**: Protects 1/3, 2/3 power points
- **Centered**: Preserves center region
- **Horizontal**: Novel "Integrity Field" for horizon preservation
- **Diagonal/Vertical**: Hough line detection
- **Triangle**: Detects 3 salient vertices
- **Curves**: Structure tensor analysis
- **Radial/Perspective**: RANSAC vanishing point estimation
- **Dense Pattern**: FFT-based periodic pattern detection
- **Scatter**: General saliency + edge preservation

### Performance Optimizations
- **Numba JIT**: 50-100× speedup for seam operations
- **Mixed precision**: bfloat16 for 2× faster depth inference
- **Energy downsampling**: Simulate costs at 256px, then interpolate

---

## Usage

### Single Image

```python
# In retargetme.ipynb
IMG_PATH = "data/retargetme/images/bike.png"
result = pipeline.run_width(
    img_path=IMG_PATH,
    target_width_ratio=0.5,  # 50% width
    manual_class=None        # Auto-detect composition
)

print(f"Class: {result['pred_class_name']}")
print(f"Operations: {result['log']['ops']}")
```

### Batch Processing

The last cell in `retargetme.ipynb` processes all RetargetMe images automatically.

**Output format**:
- `results/retargetme/<image>_width50.png` - Retargeted image
- `results/retargetme/<image>_width50_meta.json` - Metadata (class, operations, timing)

---

## Training DINOv3 Classifier (Optional)

### Download PICD Dataset
```bash
mkdir -p data/picd
gdown --id 1rqI1qT_WVx8c9iE9TeC_X3d__875NBQp -O data/picd/PICD.zip
cd data/picd && unzip PICD.zip && cd ../..
```

### Run Training
```bash
jupyter lab
# Open notebooks/train_dinov3_picd.ipynb and run all cells
```

**Configuration**:
- Dataset: PICD (~43K images, 24→10 classes)
- Architecture: Frozen DINOv3 backbone + linear head
- Validation accuracy: ~75-80%
- Output: `checkpoints/dinov3/best_dinov3_classifier_reduced.pth`

---

## Requirements

- **Python**: 3.9+
- **GPU**: NVIDIA with CUDA (8GB+ VRAM recommended)

### Key Dependencies
```
torch==2.5.1
torchvision==0.20.1
transformers==4.57.1
opencv-python==4.11.0.86
numpy==1.26.4
scipy==1.15.3
numba==0.62.1
gdown
tqdm==4.66.4
```

---

## Performance

**Processing Time** (700×466 → 350×466, A40 GPU):
- Model inference: ~450ms
- Seam carving: ~310ms
- **Total**: ~760ms/image
---

## References

- **PICD Dataset**: [CVPR 2025 Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Can_Machines_Understand_Composition_Dataset_and_Benchmark_for_Photographic_Image_CVPR_2025_paper.pdf)
- **DINOv3**: [GitHub](https://github.com/facebookresearch/dinov2)
- **Depth Anything 3**: [GitHub](https://github.com/ByteDance-Seed/depth-anything-3)
- **RGB-VST**: [GitHub](https://github.com/nnizhang/VST)
- **RetargetMe**: [Project Page](https://people.csail.mit.edu/mrub/retargetme/)

---

## License

MIT License. External models have their own licenses - see respective repositories.
