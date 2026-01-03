import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import ViTImageProcessor

# ----------------------------
# Config
# ----------------------------
ONNX_MODEL_PATH = "C:/Users/Mina/Downloads/emotion_vit_model_quantized.onnx"

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ----------------------------
# Load ONNX model
# ----------------------------
session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ----------------------------
# Load ViT image processor
# ----------------------------
processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

# ----------------------------
# Face detector
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Convert to PIL (RGB)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        # Preprocess (ViT standard)
        inputs = processor(images=face_pil, return_tensors="np")

        # ONNX inference
        ort_inputs = {input_name: inputs["pixel_values"]}
        logits = session.run([output_name], ort_inputs)[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[0][pred_class])

        # Draw
        label = f"{class_names[pred_class]} {confidence:.2f}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    cv2.imshow("Emotion Recognition (ONNX)", frame)

    if cv2.waitKey(1) & 0xFF == ord("d"):
        break

cap.release()
cv2.destroyAllWindows()

