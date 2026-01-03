import numpy as np
import onnxruntime as ort


def load_onnx_model(onnx_model_path):
    session = ort.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return session, input_name, output_name


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def predict(session, input_name, output_name, image_tensor, class_names):
    if not isinstance(image_tensor, np.ndarray):
        raise TypeError("image_tensor must be a NumPy array")

    outputs = session.run(
        [output_name],
        {input_name: image_tensor}
    )

    logits = outputs[0]
    probs = softmax(logits)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(probs[0, pred_idx])
    emotion = class_names[pred_idx]

    return emotion, confidence, probs
