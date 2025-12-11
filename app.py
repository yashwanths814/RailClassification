import io
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, redirect

# ================== FIREBASE / FIRESTORE ==================
import firebase_admin
from firebase_admin import credentials, firestore

# --------- CHANGE THIS PATH FOR RENDER ----------
# 1. Download service account JSON from Firebase console
# 2. Add it to your repo (e.g. "serviceAccountKey.json") or mount via env
# 3. Put the correct path here
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"

# Initialize Firebase Admin only once
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)
fs_client = firestore.client()

# ================== CONFIG ==================

# ⚠️ CHANGE THIS PATH ON RENDER:
#   Put best.onnx inside your repo, e.g. in "model/best.onnx"
#   Then use: MODEL_PATH = "model/best.onnx"
MODEL_PATH = r"model/best.onnx"  # <-- update for Render

# EXACTLY as in railway_station/data.yaml
CLASS_NAMES = [
    "clip_ok",
    "liner_ok",
    "pad_ok",
    "sleeper_ok",
    "bolt_ok",
    "erc_ok",
    "clip_faulty",
    "liner_faulty",
    "pad_faulty",
    "sleeper_faulty",
    "bolt_faulty",
    "erc_faulty",
    "clip_rust",
    "liner_rust",
    "pad_rust",
    "sleeper_rust",
    "bolt_rust",
    "erc_rust",
    "clip_missing",
    "liner_missing",
    "pad_missing",
    "sleeper_missing",
    "bolt_missing",
    "erc_missing",
    "qr_code",
]

CONF_THRESHOLD = 0.35
IMG_SIZE = 640

# ================== FLASK APP ==================

app = Flask(__name__)

# Load ONNX session once
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


# ================== HELPERS ==================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_class(name: str):
    """
    Map any of:
      erc_ok / erc_faulty / erc_rust / erc_missing → "erc"
      liner_* → "liner", etc.
      qr_code → None
    """
    name = name.lower()
    for comp in ["erc", "liner", "sleeper", "clip", "pad", "bolt"]:
        if comp in name:
            return comp
    return None


def preprocess_image(image_bgr, img_size=IMG_SIZE):
    """
    BGR -> RGB, resize, normalize, HWC->CHW, add batch.
    """
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_batched = np.expand_dims(img_chw, axis=0)
    return img_batched


def run_onnx_inference(image_bgr):
    """
    Returns a SINGLE best detection as:
      { "component": "erc" | "liner" | ..., "conf": float }
    or None if nothing above threshold.
    """
    input_name = session.get_inputs()[0].name
    inp = preprocess_image(image_bgr)

    outputs = session.run(None, {input_name: inp})
    preds = outputs[0]

    # Remove batch dim → [C,N] or [N,C]
    preds = np.squeeze(preds)

    if preds.ndim != 2:
        raise RuntimeError(f"Unexpected ONNX output shape: {preds.shape}")

    num_classes = len(CLASS_NAMES)
    expected_c = 4 + num_classes  # 4 box + nc class scores

    # Normalize to [N, 4+nc]
    if preds.shape[1] == expected_c:
        pred_nc = preds                     # [N, 4+nc]
    elif preds.shape[0] == expected_c:
        pred_nc = preds.T                   # [4+nc, N] -> [N, 4+nc]
    elif preds.T.shape[1] == expected_c:
        pred_nc = preds.T
    else:
        raise RuntimeError(
            f"Cannot interpret ONNX output shape {preds.shape} for 4+{num_classes} classes"
        )

    # First 4: boxes (unused here), rest: scores
    boxes = pred_nc[:, :4]
    scores_raw = pred_nc[:, 4:]

    # If scores_raw > 1.0, it's logits -> apply sigmoid
    if scores_raw.max() > 1.0:
        scores = sigmoid(scores_raw)
    else:
        scores = scores_raw

    # Best class and score per row
    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), class_ids]

    # Filter low-confidence
    mask = confs >= CONF_THRESHOLD
    if not np.any(mask):
        return None

    class_ids = class_ids[mask]
    confs = confs[mask]
    boxes = boxes[mask]

    # Single best detection
    best_idx = int(np.argmax(confs))
    best_cls = int(class_ids[best_idx])
    best_conf = float(confs[best_idx])

    if best_cls < 0 or best_cls >= num_classes:
        return None

    class_name = CLASS_NAMES[best_cls]
    component = parse_class(class_name)

    # Debug print to console
    print(
        f"[ONNX] best_idx={best_idx}, cls_id={best_cls}, "
        f"class='{class_name}', component='{component}', conf={best_conf:.4f}"
    )

    if component is None:
        return None

    return {
        "component": component,
        "conf": best_conf,
    }


def save_result_to_firestore(material_id: str, component: str, conf: float):
    """
    Update Firestore document:
      materials/{material_id}
    with AI verification fields.
    """
    try:
        doc_ref = fs_client.collection("materials").document(material_id)
        doc_ref.set(
            {
                "aiVerified": True,
                "aiVerifiedComponent": component.upper(),
                "aiVerifiedConfidence": conf,
                "aiVerifiedAt": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        print(f"[Firestore] Updated materials/{material_id} with AI verification.")
    except Exception as e:
        print("[Firestore] Error updating document:", e)


def render_page(message_html: str = "", material_id: str | None = None):
    """
    Simple HTML page with file upload and result area.
    `message_html` is injected into the result div.
    If material_id is present, we show it and embed in a hidden input.
    """
    if material_id:
        material_info_html = f"""
      <p style="font-size:0.8rem;color:#444;margin-bottom:8px;">
        Material ID: <b>{material_id}</b>
      </p>
      <input type="hidden" name="materialId" value="{material_id}" />
        """
    else:
        material_info_html = ""

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Track Fitting AI Verification</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(135deg, #F7E8FF, #FDFBFF, #E4D4FF);
      min-height: 100vh;
      margin: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .card {{
      background: rgba(255, 255, 255, 0.9);
      border-radius: 20px;
      padding: 20px 24px;
      box-shadow: 0 15px 35px rgba(0,0,0,0.08);
      max-width: 420px;
      width: 100%;
    }}
    h1 {{
      margin-top: 0;
      font-size: 1.4rem;
      color: #4B3A7A;
    }}
    p {{
      font-size: 0.85rem;
      color: #555;
    }}
    .file-input {{
      margin: 12px 0;
    }}
    input[type="file"] {{
      font-size: 0.8rem;
    }}
    button {{
      margin-top: 10px;
      width: 100%;
      padding: 10px 14px;
      border-radius: 999px;
      border: none;
      background: #A259FF;
      color: white;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
    }}
    button:hover {{
      background: #8E3FE8;
    }}
    .result {{
      margin-top: 16px;
      font-size: 0.9rem;
      padding: 10px 12px;
      border-radius: 12px;
      background: #F7E8FF;
      color: #4B3A7A;
    }}
    .error {{
      background: #FEE2E2;
      color: #991B1B;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Track Fitting Element Verification</h1>
    <p>Upload a Jio-tag or track fitting image. The AI will identify whether it is ERC, Liner, Pad, Sleeper, Clip or Bolt.</p>
    <form method="POST" action="/verify" enctype="multipart/form-data">
      {material_info_html}
      <div class="file-input">
        <input type="file" name="image" accept="image/*" required />
      </div>
      <button type="submit">Run AI Check</button>
    </form>

    <div class="result">
      {message_html if message_html else "No image analyzed yet."}
    </div>
  </div>
</body>
</html>
"""


# ================== ROUTES ==================

@app.route("/")
def root():
    # Redirect root to /verify
    return redirect("/verify")


@app.route("/verify", methods=["GET", "POST"])
def verify():
    if request.method == "GET":
        # Optional materialId from query param: /verify?materialId=ABC1234
        material_id = request.args.get("materialId")
        return render_page(material_id=material_id)

    # POST: handle image upload and run inference
    file = request.files.get("image")
    material_id = request.form.get("materialId")  # may be None

    if not file or file.filename == "":
        return render_page('<span class="error">No image selected.</span>', material_id)

    # Read file into OpenCV BGR image
    try:
        file_bytes = file.read()
        file_array = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Decode error:", e)
        return render_page('<span class="error">Could not read image data.</span>', material_id)

    if img_bgr is None:
        return render_page('<span class="error">Failed to decode image.</span>', material_id)

    try:
        det = run_onnx_inference(img_bgr)
    except Exception as e:
        print("Inference error:", e)
        return render_page(
            f'<span class="error">Inference error: {str(e)}</span>',
            material_id,
        )

    if det is None:
        return render_page(
            "No ERC/LINER/PAD/SLEEPER/CLIP/BOLT detected above threshold.",
            material_id,
        )

    comp = det["component"].upper()
    conf = det["conf"]

    # If we have a materialId, update Firestore
    if material_id:
        save_result_to_firestore(material_id, comp, conf)

    msg = f"<b>Detected:</b> {comp} &nbsp; <b>Confidence:</b> {conf:.2f}"
    if material_id:
        msg += f"<br/><br/>Result stored for <b>Material ID {material_id}</b> in Firestore."

    return render_page(msg, material_id)


# ================== MAIN ==================

if __name__ == "__main__":
    # Run development server locally
    app.run(host="0.0.0.0", port=5000, debug=True)
