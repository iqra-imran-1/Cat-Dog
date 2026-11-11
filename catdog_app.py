import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

MODEL_PATH = "catdog.keras"
from keras.src.layers.core.dense import Dense

_orig_init = Dense.__init__
def _patched_init(self, *args, quantization_config=None, **kwargs):
    _orig_init(self, *args, **kwargs)
Dense.__init__ = _patched_init

model = tf.keras.models.load_model(MODEL_PATH)
app = FastAPI(title="Cat vs Dog Classifier")

HTML = """
<!DOCTYPE html>
<html>
<body style="font-family:sans-serif;max-width:420px;margin:40px auto;text-align:center">
  <h2>Cat or Dog?</h2>
  <input type="file" id="f" accept="image/*"><br><br>
  <img id="preview" style="max-width:100%;display:none"><br>
  <button onclick="send()">Classify</button>
  <h3 id="result"></h3>
<script>
const f = document.getElementById("f");
f.onchange = () => {
  const p = document.getElementById("preview");
  p.src = URL.createObjectURL(f.files[0]);
  p.style.display = "block";
};
async function send() {
  if (!f.files[0]) return;
  const fd = new FormData();
  fd.append("file", f.files[0]);
  const r = await fetch("/predict", {method: "POST", body: fd});
  const d = await r.json();
  document.getElementById("result").innerText =
    r.ok ? d.label + " (" + (d.confidence * 100).toFixed(1) + "%)" : d.detail;
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img = img.convert("RGB").resize((128, 128))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = np.expand_dims(np.array(img, dtype="float32"), 0)
    p = float(model.predict(x, verbose=0)[0][0])

    if p < 0.3:
        label, conf = "cat", 1 - p
    elif p > 0.7:
        label, conf = "dog", p
    else:
        label, conf = "uncertain (probably not a cat or dog)", max(p, 1 - p)

    return {"label": label, "confidence": round(conf, 4)}