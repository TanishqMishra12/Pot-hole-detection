# Pothole & Road Damage Detector

## Overview
This project detects and measures pothole and road-surface damage from images — the old-school way, using classical computer vision instead of deep learning. It looks at each image, figures out how much of the surface is damaged, and scores the severity as Low (<2%), Medium (2–8%), or High (>8%) based on the damage area ratio. Results are exported as structured CSV/JSON reports, making it easy to review a batch of images at a glance.

## Features
- **Classical Segmentation Engine**: Choose between a blazing-fast Adaptive Threshold method or a robust MRF Graph-Cut method.
- **Automated Severity Scoring**: Calculates the damage area ratio and classifies severity into Low (<2%), Medium (2-8%), and High (>8%).
- **Batch Processing & Reporting**: Processes hundreds of images at once and exports results to CSV and JSON.
- **Mask Visualization**: Optionally exports images with damage highlighted as red overlays for easy verification.
- **IoU Evaluation**: Automatically parses Pascal VOC XML annotations to compute Intersection over Union metrics against ground truth.

## Methodology

### Preprocessing Pipeline
To prepare the images for segmentation, a standard pipeline is applied:
1. **Grayscale Conversion**: Reduces dimensionality.
2. **Denoising**: Applies a Gaussian blur (or optional Median blur) to remove high-frequency noise while preserving edges.
3. **Contrast Enhancement**: Utilizes CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize lighting conditions and highlight texture differences on the road surface.

### Method A: Adaptive Thresholding
A rapid baseline approach utilizing morphological operations:
- Applies an adaptive Gaussian threshold to isolate dark damage regions.
- Uses morphological Opening and Closing (with an elliptical kernel) to remove isolated noise pixels and fill small holes in detected potholes.

### Method B: MRF Graph-Cut Segmentation
An advanced energy minimization approach utilizing a Markov Random Field (MRF):
- **Data Term (NLL)**: The pipeline uses the adaptive threshold result as a "seed" to estimate Gaussian statistics for "road" vs "pothole" pixels. We compute the Negative Log-Likelihood (NLL) of each pixel, heavily penalized by class priors to prevent over-segmentation.
- **Smoothness Term (Ising Model)**: We implement a contrast-sensitive Ising model that penalizes label changes across regions of similar intensity but allows cuts at strong edges (high intensity gradients).
- **Optimization**: The energy function is minimized by constructing a capacity graph and solving for the min-cut/max-flow using `scipy.sparse.csgraph.maximum_flow`.

## Project Structure
```text
pothole-detector/
├── main.py              # CLI entry point & batch orchestration
├── config.py            # Configuration dataclass (tunable parameters)
├── preprocessing.py     # Denoising and CLAHE contrast enhancement
├── segmentation.py      # Threshold & MRF Graph-Cut segmentation logic
├── analysis.py          # Severity scoring, IoU evaluation, & exporting
├── utils.py             # I/O, resizing, annotation parsing, logging
├── requirements.txt     # Python dependencies
└── tests/               # Pytest suite (Unit & Integration tests)
```

## Configuration & Tuning
The pipeline is highly modular and tunable via `config.py`. Key parameters include:
- `graphcut_lambda`: Controls the relative weight of the spatial smoothness term (higher values produce smoother, more contiguous blobs).
- `min_contour_area`: Filters out tiny, negligible detections that are likely noise.
- `morph_kernel`: Controls the size of the morphological structuring element used for cleanup.

## Evaluation & Limitations

### Metric: Intersection over Union (IoU)
Since this pipeline produces pixel-level segmentation masks but evaluates against Pascal VOC bounding boxes, the system extracts the extreme contours of the generated masks, draws a bounding box around them, and computes the IoU against the ground-truth annotations.

### Limitations
As a purely classical CV approach, the system is sensitive to extreme lighting variations. Heavy, sharp shadows cast by trees or vehicles on the road can sometimes fool the data term into classifying dark shadows as potholes. This is an expected trade-off when bypassing computationally heavy deep-learning feature extractors.

## Technologies/Tools Used
- **Language**: Python 3.13
- **Computer Vision**: OpenCV (`cv2`) for image processing, filtering, CLAHE, and thresholding.
- **Math & Optimization**: NumPy for matrix operations, SciPy for Graph-Cut energy minimization.
- **Testing**: `pytest` for unit and integration testing.

## Installation & Setup

1. **Clone and navigate to the directory**:
   ```bash
   git clone https://github.com/TanishqMishra12/Pot-hole-detection.git
   cd Pot-hole-detection/pothole-detector
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage & Execution

Run the pipeline using the CLI entry point `main.py`. 

**Basic Run (Fast Threshold Method):**
```bash
python main.py
```

**Graph-Cut Method on 10 images with Overlays:**
```bash
python main.py --method graphcut --max-images 10 --save-masks
```

**Custom Directories:**
```bash
python main.py --input /path/to/images --output /path/to/results --annotations /path/to/annotations
```

## Testing Instructions

The project includes a comprehensive test suite for preprocessing, segmentation, and analysis modules.
Run the tests using pytest:
```bash
pytest tests/ -v
```

## Example Outputs (Screenshots)

Below are the batch summaries generated by the CLI for both methods on a dataset of 665 images.

### Graph-Cut Method Execution
<img width="770" height="292" alt="image" src="https://github.com/user-attachments/assets/7abacea7-b4a9-46d1-b365-5ff2b50c469e" />

### Threshold Method Execution
<img width="797" height="260" alt="image" src="https://github.com/user-attachments/assets/cb4a9cda-934a-4853-970e-b962450e11e5" />
