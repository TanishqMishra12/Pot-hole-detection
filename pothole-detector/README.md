# Pothole / Road Damage Detector

**Course:** Computer Vision (24BAI10886)  
**Type:** Individual Project — Classical CV, CLI-based  

## Overview

A classical computer-vision pipeline that automatically detects and quantifies pothole / road-surface damage from images, producing structured severity reports (CSV/JSON) — no deep learning, no GUI.

## Setup

```bash
cd pothole-detector
pip install -r requirements.txt
```

**Dataset:** Place images in `../images/` and annotations in `../annotations/` (or specify paths via CLI flags).

## Usage

```bash
# Default: adaptive threshold method on all images
python main.py

# Graph-Cut (MRF) method on first 10 images, with mask overlays
python main.py --method graphcut --max-images 10 --save-masks

# Custom paths
python main.py --input /path/to/images --output /path/to/results --annotations /path/to/annotations

# Quick timing test (recommended before full batch)
python main.py --method graphcut --max-images 5 --save-masks
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--input` | `../images` | Input image directory |
| `--output` | `results/` | Output directory |
| `--method` | `threshold` | `threshold` or `graphcut` |
| `--annotations` | `../annotations` | Pascal VOC XML annotation directory |
| `--save-masks` | off | Save mask overlay images |
| `--max-images` | all | Limit number of images |
| `--graphcut-resolution` | 400 | Max resolution for Graph-Cut (lower = faster) |

## Output

```
results/
├── report.csv          # Per-image: filename, area ratio, severity, IoU, timing
├── report.json         # Same data in JSON format
├── masks/              # Overlay images (if --save-masks)
└── pipeline.log        # Full debug log
```

## Segmentation Methods

### Method A: Adaptive Thresholding (default)
Fast baseline — adaptive Gaussian threshold with morphological open/close to isolate dark damage regions.

### Method B: Graph-Cut / MRF
Energy minimization over a Markov Random Field:
- **Data term:** Gaussian NLL from Otsu-split intensity distributions
- **Smoothness term:** Contrast-sensitive Ising model
- **Solver:** scipy `maximum_flow` (min-cut/max-flow)

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
pothole-detector/
├── main.py              # CLI entry point
├── config.py            # All tunable parameters
├── preprocessing.py     # Grayscale → denoise → CLAHE
├── segmentation.py      # Threshold + Graph-Cut methods
├── analysis.py          # Area/severity scoring, IoU eval, reporting
├── utils.py             # I/O, annotation parsing, logging
├── requirements.txt
├── README.md
└── tests/
    ├── test_preprocessing.py
    ├── test_segmentation.py
    ├── test_analysis.py
    └── test_integration.py
```

## References

- [Pothole Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/pothole-detection)
- Course notes: Image Segmentation (Graph-Cut, Mean-Shift, MRF), Histogram Processing
