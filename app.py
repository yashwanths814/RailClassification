import os
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "best.onnx"))
IMG_SIZE = 640
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))

CLASS_NAMES = [
    "clip_ok","liner_ok","pad_ok","sleeper_ok","bolt_ok","erc_ok",
    "clip_faulty","liner_faulty","pad_faulty","sleeper_faulty","bolt_faulty","erc_faulty",
    "clip_rust","liner_rust","pad_rust","sleeper_rust","bolt_rust","erc_rust",
    "clip_missing","liner_missing","pad_missing","sleeper_missing","bolt_missing","erc_missing",
    "qr_code",
]

app = Flask(__name__)
CORS(app)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def parse_component(class_name: str):
    n = class_name.lower()
    for comp in ["erc", "liner", "sleeper", "clip", "pad", "bolt"]:
        if comp in n:
            return comp
    return None

def preprocess_image(image_bgr):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def run_onnx_inference_yolo(image_bgr):
    input_name = session.get_inputs()[0].name
    inp = preprocess_image(image_bgr)
    out = session.run(None, {input_name: inp})[0]
    preds = np.squeeze(out)

    if preds.ndim != 2:
        raise RuntimeError(f"Unexpected ONNX output shape: {preds.shape}")

    nc = len(CLASS_NAMES)
    expected_c = 4 + nc

    if preds.shape[1] == expected_c:
        p = preds
    elif preds.shape[0] == expected_c:
        p = preds.T
    elif preds.T.shape[1] == expected_c:
        p = preds.T
    else:
        raise RuntimeError(f"Cannot interpret ONNX output shape {preds.shape} (expected {expected_c})")

    scores_raw = p[:, 4:]
    scores = sigmoid(scores_raw) if scores_raw.max() > 1.0 else scores_raw

    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), class_ids]

    mask = confs >= CONF_THRESHOLD
    if not np.any(mask):
        return None

    class_ids = class_ids[mask]
    confs = confs[mask]

    best_i = int(np.argmax(confs))
    best_cls = int(class_ids[best_i])
    best_conf = float(confs[best_i])

    if best_cls < 0 or best_cls >= len(CLASS_NAMES):
        return None

    class_name = CLASS_NAMES[best_cls]
    component = parse_component(class_name)
    if component is None:
        return None

    return {"component": component, "confidence": best_conf}

@app.route("/verify", methods=["GET", "POST"])
def verify():
    if request.method == "GET":
        return send_file(os.path.join(BASE_DIR, "index.html"))

    file = request.files.get("image")
    if not file:
        return jsonify({"ok": False, "error": "image is required"}), 400

    img_bgr = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"ok": False, "error": "failed to decode image"}), 400

    det = run_onnx_inference_yolo(img_bgr)

    if det is None:
        return jsonify({
            "ok": True,
            "component": None,
            "confidence": 0,
            "confidencePercent": 0.0
        })

    conf_percent = float(det["confidence"]) * 100.0
    return jsonify({
        "ok": True,
        "component": det["component"],
        "confidence": float(det["confidence"]),
        "confidencePercent": conf_percent
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
