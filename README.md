# Human Emotion Recognition using Vision Transformers

This project implements a **human emotion recognition system** using a **Vision Transformer (ViT)** trained on the FER2013 dataset.  
The work covers dataset handling, class imbalance mitigation, transfer learning, advanced data augmentation, evaluation beyond aggregate accuracy, model quantization, and real-time inference.

The focus is on **robust model behavior, transparent evaluation, and deployment readiness**, rather than treating emotion recognition as a purely academic benchmark.


## Problem Overview

Facial emotion recognition is a challenging computer vision task due to:
- Subtle inter-class differences
- High intra-class variability
- Significant class imbalance
- Sensitivity to pose, lighting, and facial expressions

This project formulates emotion recognition as a **7-class image classification problem**, aiming to produce a model that generalizes well and behaves reliably under real-world conditions.


## Dataset

The project uses the FER2013 dataset for multi-class facial emotion recognition across seven emotion categories. Images are provided as grayscale facial samples and are dynamically loaded from the original CSV format without storing raw data in the repository. The dataset’s official training, validation, and test splits are preserved to ensure reproducibility. Detailed access instructions are available in data/README.md


## Preprocessing & Data Augmentation

To match Vision Transformer input requirements and improve generalization, the following preprocessing pipeline is applied:

- Conversion from grayscale to RGB
- Resize to **224 × 224**
- ImageNet normalization
- Augmentation applied **only to the training split**, including:
  - Random resized cropping
  - Horizontal flipping
  - **MixUp augmentation** to improve robustness and reduce overfitting

All preprocessing logic is centralized and reproducible.


## Class Imbalance Handling

The FER2013 dataset exhibits **severe class imbalance**, particularly for minority classes such as *Disgust*.

To address this:
- Class frequencies are computed from the training split
- Inverse-frequency weights are derived
- A **WeightedRandomSampler** is applied **only to the training DataLoader**

Validation and test sets remain unaltered to ensure unbiased evaluation.


## Model Architecture

The model is based on **transfer learning with a pretrained Vision Transformer**:

- Backbone: `google/vit-base-patch16-224-in21k`
- Modified classification head:
  - Dropout
  - Linear projection to 7 emotion classes
- Selective fine-tuning:
  - ViT embeddings and first encoder layer are frozen
  - Higher encoder layers and classifier are trainable

This design balances representation quality, training stability, and computational efficiency.


## Training Strategy

Key training decisions include:
- Transfer learning with layer freezing
- Class-balanced sampling during training
- Separate learning rates for backbone and classifier
- Hybrid learning rate schedule:
  - Linear warmup
  - Plateau phase
  - Cosine decay
- Early stopping based on validation loss

Training and evaluation are conducted in structured notebooks to allow detailed inspection and debugging.


## Evaluation & Metrics

Model performance is evaluated using a combination of **aggregate and class-level metrics**, including:
- Overall accuracy
- Confusion matrices
- ROC-style analysis (where applicable)
- **Classification reports**, providing per-class:
  - Precision
  - Recall
  - F1-score

This ensures that minority-class behavior is explicitly analyzed, rather than hidden behind global accuracy metrics.


## Quantization & Optimization

To enable efficient deployment, the trained model is exported to ONNX format and optimized using dynamic quantization. This reduces model size and improves CPU inference performance while preserving accuracy, without requiring retraining. The full conversion and optimization workflow is documented in notebooks/Quantization.ipynb.


## Real-Time Inference

A real-time emotion recognition pipeline is implemented using OpenCV for webcam capture and face detection, combined with ONNX Runtime for fast CPU-based inference. Each detected face is processed using Vision Transformer–compatible preprocessing, producing an emotion prediction along with a confidence score. This demonstrates practical deployment of a transformer-based vision model in a live setting.


## License

This project is released under the **MIT License**, allowing reuse with attribution.


## Author

**Mina Nagy**
