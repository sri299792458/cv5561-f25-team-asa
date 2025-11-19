# Class-Aware Seam Carving 2025 (RetargetMe)

This repo implements a **modern, class-aware content-aware image retargeting pipeline** for the **RetargetMe** benchmark.

### Core components:

-   **DINOv3** classifier (10 composition classes)
    
-   **Depth Anything 3 (DA3)** for monocular depth + confidence
    
-   **Visual Saliency Transformer (RGB-VST)** for learned saliency
    
-   **Class-aware energy** combining:
    
    -   image gradients
        
    -   depth edges
        
    -   VST saliency
        
    -   composition priors from DINOv3
        
-   **Multi-operator retargeting**: crop + scale + depth-aware seam carving (instead of pure seam deletion)
    

The main Jupyter notebook:

1.  Loads a RetargetMe image
    
2.  Runs DINOv3 + DA3 + VST
    
3.  Builds the energy map
    
4.  Applies multi-operator retargeting
    
5.  Shows visual outputs after each step (saliency, depth, `|∇I|`, `|∇D|`, class term `T`, final energy `E`, final image)
    

## 1. Repo Layout

Recommended structure:

```
your-repo/
├─ README.md
├─ requirements.txt
├─ notebooks/
│  └─ retargetme_vst_da3.ipynb
│  └─ train_dinov3_picd.ipynb       # (optional) DINOv3 training on PICD
├─ checkpoints/
│  └─ dinov3/
│    └─ best_dinov3_classifier_reduced.pth
├─ RGB_VST/             # unzipped from Google Drive
├─ depth-anything-3/    # DA3 repo (installed with pip -e .)
├─ data/
│  └─ retargetme/
│    └─ images/         # RetargetMe images
└─ results/
   └─ retargetme/       # outputs written by the notebook

```

Run Jupyter from the repo root:

```
cd your-repo
jupyter lab
# or
jupyter notebook

```

_All paths in the notebook are relative to the repo root._

## 2. Requirements

-   **OS:** Linux (recommended) or other UNIX-like
    
-   **Python:** 3.9+ (tested with 3.9 / 3.10)
    
-   **GPU:** NVIDIA with CUDA (strongly recommended)
    
-   **CUDA/cuDNN:** compatible with your PyTorch install
    

### Python packages

Install via pip / `requirements.txt`:

-   `torch`
    
-   `torchvision`
    
-   `transformers`
    
-   `opencv-python`
    
-   `numpy`
    
-   `scipy`
    
-   `Pillow`
    
-   `matplotlib`
    
-   `numba`
    
-   `gdown`
    
-   `tqdm` (optional but useful)
    

Install everything with:

```
pip install -r requirements.txt

```

## 3. Environment Setup

### 3.1 Clone this repo

```
git clone <this-repo-url>
cd <this-repo-folder>

```

(Optional, but recommended) Create a conda environment:

```
conda create -n seam2025 python=3.10 -y
conda activate seam2025

```

Then install requirements:

```
pip install -r requirements.txt

```

### 3.2 Install Depth Anything 3 (DA3)

```
git clone [https://github.com/ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3)
cd depth-anything-3
pip install -e . --no-deps
cd ..

```

This makes `depth_anything_3` importable in the notebook.

### 3.3 Hugging Face token (optional but recommended)

Some models (DA3 / DINOv3 backbone) are pulled from Hugging Face. Set your token in the terminal:

```
export HUGGING_FACE_HUB_TOKEN="your_hf_token_here"

```

The notebook reads this variable:

```
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)

```

_If you don’t need private models, you can omit this._

## 4. Checkpoints & External Code

### 4.1 DINOv3 Composition Head (in repo)

The DINOv3 classifier head checkpoint is tracked directly in this repo (tiny, ~96 KB).

**Expected location:**

```
checkpoints/
  dinov3/
    best_dinov3_classifier_reduced.pth

```

In the notebook:

```
CKPT_PATH = "checkpoints/dinov3/best_dinov3_classifier_reduced.pth"

```

**At runtime:**

1.  The backbone `facebook/dinov3-vitb16-pretrain-lvd1689m` is loaded from Hugging Face using `AutoModel.from_pretrained(...)`.
    
2.  The `.pth` file here only contains the 10-way head weights, loaded into `model.head`.
    

### 4.2 RGB-VST (code + weights as a zip from Drive)

The entire `RGB_VST` folder (Python files + `RGB_VST.pth` weights) is stored as a zip on Google Drive.

From the repo root, download and unpack it (replace `<RGB_VST_ZIP_FILE_ID>` with your actual ID):

```
gdown --id <RGB_VST_ZIP_FILE_ID> -O RGB_VST.zip
unzip RGB_VST.zip

```

You should end up with:

```
RGB_VST/
  Models/
    ImageDepthNet.py
    ...
  checkpoint/
    RGB_VST.pth
  pretrained_model/
    80.7_T2T_ViT_t_14.pth.tar
  ...

```

The notebook assumes:

```
import sys

rgb_vst_path = "RGB_VST"
if rgb_vst_path not in sys.path:
    sys.path.insert(0, rgb_vst_path)

from Models.ImageDepthNet import ImageDepthNet

```

_If your zip expands to a different folder name, either rename it to `RGB_VST` or adjust the path in the notebook._

### 4.3 Training the DINOv3 Classifier on PICD (optional)

This repo also includes a **training notebook** for the DINOv3 composition classifier, based on the **[PICD]([CV-xueba/PICD_ImageComposition](https://github.com/CV-xueba/PICD_ImageComposition)) (Photographic Image Composition Dataset)**:

- [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Can_Machines_Understand_Composition_Dataset_and_Benchmark_for_Photographic_Image_CVPR_2025_paper.pdf): _“Can Machines Understand Composition? Dataset and Benchmark for Photographic Image Composition (PICD)”_, CVPR 2025.
    
        

Training notebook location:

`notebooks/train_dinov3_picd.ipynb` 

You can optionally download the PICD dataset via `gdown`:

`mkdir -p data/picd
gdown --id 1rqI1qT_WVx8c9iE9TeC_X3d__875NBQp -O data/picd/PICD.zip cd data/picd
unzip PICD.zip cd ../..` 

The training notebook expects the PICD data under `data/picd/` (you can adjust paths inside the notebook if needed).

Once trained, the resulting classifier head can be saved to:

`checkpoints/dinov3/best_dinov3_classifier_reduced.pth` 

and will be picked up automatically by `retargetme_vst_da3.ipynb` for inference.

## 5. RetargetMe Dataset

Download RetargetMe images:

```
mkdir -p data/retargetme
cd data/retargetme
wget [https://people.csail.mit.edu/mrub/retargetme/download/images-20100824.zip](https://people.csail.mit.edu/mrub/retargetme/download/images-20100824.zip)
unzip images-20100824.zip
cd ../..

```

Expected structure:

```
data/
  retargetme/
    images/
      bike.png
      ...

```

In the notebook, for a single debug image:

```
IMG_PATH = "data/retargetme/images/bike.png"

```

Later, a batch cell loops over all images in `data/retargetme/images`.

## 6. Running the Notebook

From the repo root:

```
jupyter lab
# or
jupyter notebook

```

Open: `notebooks/retargetme_vst_da3.ipynb`

**The notebook structure:**

1.  **Imports & utilities**
    
2.  **VST saliency setup**
    
    -   adds RGB_VST to sys.path
        
    -   loads RGB_VST.pth
        
3.  **DINOv3 classifier**
    
    -   backbone from Hugging Face
        
    -   head from `checkpoints/dinov3/best_dinov3_classifier_reduced.pth`
        
4.  **DA3 depth wrapper**
    
5.  **Geometry utilities** (Hough lines, structure tensor, etc.)
    
6.  **Energy construction** using:
    
    -   image gradient `|∇I|`
        
    -   depth edge `|∇D|`
        
    -   VST saliency
        
    -   class-specific term `T` based on composition
        
7.  **Multi-operator retargeting** (crop + scale + depth-aware seam carving)
    
8.  **Pipeline wrapper** (`RetargetPipeline`)
    
9.  **Single-image debug cell** (visualizing intermediate maps and the final output)
    
10.  **Batch cell** to retarget all RetargetMe images
    

## 7. Batch RetargetMe Evaluation

The last cell of the notebook loops over all images in `data/retargetme/images`, runs the full pipeline, and writes results into:

```
results/retargetme/
  <image_stem>_<direction><keep>.png
  <image_stem>_<direction><keep>_meta.json

```

Where, for example:

-   `<direction>` = width or height
    
-   `<keep>` = e.g. 50 for `KEEP_RATIO = 0.50`
    

The metadata JSON includes:

-   input/output paths
    
-   original and target sizes
    
-   direction and keep ratio
    
-   predicted composition class + probabilities
    
-   last few operations (crop/scale/seam) in the log
    
-   runtime per image
    

You can then use those outputs for your evaluation on RetargetMe (metrics, user study, etc.).

## 8. Notes

-   **Performance:** The pipeline is designed for GPU; running DA3 + VST + DINOv3 on CPU will be very slow.
    
-   **Numba:** Used to accelerate seam carving; the notebook warms up the JIT once at initialization.
    
-   **Path Changes:** If you change folder names or move the notebook, just update:
    
    -   `CKPT_PATH`
        
    -   `rgb_vst_path`
        
    -   dataset paths like `IMG_PATH` and `data/retargetme/images`