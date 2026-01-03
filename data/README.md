# FER2013 Dataset

This project uses the **FER2013 (Facial Expression Recognition 2013)** dataset for training and evaluating a facial emotion recognition model.

No raw dataset files are included in this repository.


## Dataset Overview

- **Dataset name:** FER2013
- **Task:** Multi-class facial emotion classification
- **Number of classes:** 7  
  - Angry  
  - Disgust  
  - Fear  
  - Happy  
  - Sad  
  - Surprise  
  - Neutral
- **Image format:** Grayscale facial images
- **Image size:** 48 × 48 pixels
- **Label format:** Integer class indices (0–6)


## Data Source

The dataset is publicly available on Kaggle:

- **FER2013 on Kaggle:**  
  https://www.kaggle.com/datasets/deadskull7/fer2013

The dataset is provided as a single CSV file (`fer2013.csv`) containing:
- `emotion` → class label
- `pixels` → space-separated pixel values
- `Usage` → predefined data split indicator


## Dataset Splits

The dataset includes predefined splits:
- **Training**
- **PublicTest** (used as validation)
- **PrivateTest** (used as test)

These splits are preserved exactly as defined in the original dataset.


## Data Loading

The dataset is loaded dynamically during experimentation and training using:
- `pandas` for CSV parsing
- A custom PyTorch `Dataset` implementation
- PyTorch `DataLoader` for batching

No extracted images, intermediate files, or preprocessing artifacts are stored in this repository.


## Notes

- Images are converted to RGB and resized to 224×224 during preprocessing to match Vision Transformer (ViT) input requirements
- Dataset handling logic is implemented in `src/dataset.py`
- All preprocessing steps are fully reproducible


## Notes

- Images are converted to RGB and resized to 224×224 during preprocessing to match Vision Transformer (ViT) input requirements
- Dataset handling logic is implemented in `src/dataset.py`
- All preprocessing steps are fully reproducible

