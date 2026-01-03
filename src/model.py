import torch.nn as nn
from transformers import ViTForImageClassification

from .config import NUM_CLASSES


def build_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=NUM_CLASSES,
    )

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )

    for param in model.vit.embeddings.parameters():
        param.requires_grad = False

    for layer in model.vit.encoder.layer[:1]:
        for param in layer.parameters():
            param.requires_grad = False

    return model
