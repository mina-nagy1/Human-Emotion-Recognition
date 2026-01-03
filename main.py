# import torch
# from transformers import ViTForImageClassification,ViTConfig
# from PIL import Image
# import torchvision.transforms as T
# import cv2
# import numpy as np
# import torch.nn as nn


# # ----------------------------
# # Config
# # ----------------------------
# device = torch.device("cpu")
# class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# num_classes = 7

# # Recreate config exactly like training
# config = ViTConfig(
#     #hidden_dropout_prob=0.15,
#     #attention_probs_dropout_prob=0.15,
#     num_labels=num_classes
# )

# # ----------------------------
# # Load model
# # ----------------------------
# model = ViTForImageClassification.from_pretrained(
#     "google/vit-base-patch16-224-in21k",
#     config=config,
# ).to(device)

# # Rebuild classifier EXACTLY like training
# in_features = model.classifier.in_features
# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.4),
#     nn.Linear(in_features, num_classes)
# ).to(device)

# #model.load_state_dict(torch.load("M:/Mina/best_vit_lastone_withplots_again.pth", map_location="cpu"))
# #model.load_state_dict(torch.load("M:/Mina/best_vit__with_class_weights.pth", map_location="cpu"))
# model.load_state_dict(torch.load("C:/Users/Mina/Downloads/emotion_vit_model_quantized.onnx", map_location="cpu"))


# model.to(device)
# model.eval()

# # ----------------------------
# # Preprocessing (must match training)
# # ----------------------------
# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.5, 0.5, 0.5],
#                 std=[0.5, 0.5, 0.5])
# ])

# # ----------------------------
# # Load face detector
# # ----------------------------
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # ----------------------------
# # Webcam loop
# # ----------------------------
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame = cv2.flip(frame,1)
#     # Convert to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         face = frame[y:y+h, x:x+w]

#         # Convert to PIL for transforms
#         face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
#         face_tensor = transform(face_pil).unsqueeze(0).to(device)

#         # Predict
#         with torch.no_grad():
#             outputs = model(pixel_values=face_tensor)
#             probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#             pred_class = torch.argmax(probs, dim=1).item()
#             confidence = probs[0][pred_class].item()

#         # Draw results
#         label = f"{class_names[pred_class]} {confidence:.2f}"
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


#     cv2.imshow("Emotion Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord("d"):
#         break

# cap.release()
# cv2.destroyAllWindows()



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
