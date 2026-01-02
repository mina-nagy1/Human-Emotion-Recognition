# Original FER2013 image properties
ORIGINAL_IMAGE_SIZE = 48

# Model input configuration
INPUT_IMAGE_SIZE = 224
CHANNELS = 3

# Emotion labels (FER2013 standard)
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

NUM_CLASSES = len(EMOTION_LABELS)

# Normalization (ImageNet – used for ViT)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training configuration
BATCH_SIZE = 32

# Paths
MODEL_DIR = "models"
MODEL_PATH = "models/emotion_model.onnx"
