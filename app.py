#   pip install streamlit torch torchvision git+https://github.com/openai/CLIP.git Pillow
#   streamlit run app.py

import io
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from model_clip import CLIPCombinedInteractModel

MODEL_WEIGHTS_PATH = "best_model_seed_142.pt"

CLASS_NAMES = {
    0: "Counterfeit (190)",
    1: "Genuine · Over 10 years (2800)",
    2: "Genuine · Over 15 years (3300)",
    3: "Counterfeit (560)",
}

NUM_CLASSES = 4

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ── helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPCombinedInteractModel(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()
    return model, device

def preprocess(pil_img):
    return transform(pil_img.convert("RGB")).unsqueeze(0)  # (1, 3, 224, 224)

def run_inference(model, device, full_img, black_img, white_img):
    full  = preprocess(full_img).to(device)
    black = preprocess(black_img).to(device)
    white = preprocess(white_img).to(device)
    with torch.no_grad():
        outputs, _, _, _ = model(full, black, white)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().tolist()
        pred  = int(torch.argmax(outputs, dim=1).item())
    return pred, probs

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CRP Classifier", page_icon="🍊", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.5px; }
    .stButton > button {
        background: #1a1a2e; color: #f0e6d3; border: none; border-radius: 4px;
        padding: 0.5rem 1.5rem; font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem; letter-spacing: 0.5px; cursor: pointer;
    }
    .stButton > button:hover { background: #e07b39; }
    .result-card {
        background: #1a1a2e; color: #f0e6d3; border-radius: 8px;
        padding: 1.5rem 2rem; margin-top: 1rem; font-family: 'IBM Plex Mono', monospace;
    }
    .result-card .label    { font-size: 1.6rem; font-weight: 600; color: #e07b39; }
    .result-card .sublabel { font-size: 0.85rem; opacity: 0.65; margin-top: 0.2rem; }
    .bar-wrap { margin-top: 1.2rem; }
    .bar-row  { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem; font-size:0.78rem; }
    .bar-name { width: 200px; text-align:right; opacity:0.8; }
    .bar-bg   { flex:1; background:#2e2e4a; border-radius:3px; height:14px; }
    .bar-fill { height:14px; border-radius:3px; background:#e07b39; }
    .bar-pct  { width:42px; text-align:right; opacity:0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── UI ─────────────────────────────────────────────────────────────────────────

st.markdown("# 🍊 CRP Classifier")
st.markdown(
    "**Citri Reticulatae Pericarpium** · CLIP ResNet-50 backbone  \n"
    "Upload three images of the pericarpium to classify its storage age."
)
st.divider()

st.markdown("### Upload Images")
st.caption(
    "**Full image** — whole CRP piece (exocarp side facing up)  \n"
    "**Black patch** — close-up crop of the dark exocarp surface  \n"
    "**White patch** — close-up of the inner albedo (flipped side)"
)

col1, col2, col3 = st.columns(3)
with col1:
    up_full  = st.file_uploader("Full image",  type=["jpg", "jpeg", "png"], key="full")
with col2:
    up_black = st.file_uploader("Black patch", type=["jpg", "jpeg", "png"], key="black")
with col3:
    up_white = st.file_uploader("White patch", type=["jpg", "jpeg", "png"], key="white")

# previews
if any([up_full, up_black, up_white]):
    p1, p2, p3 = st.columns(3)
    if up_full:
        p1.image(Image.open(up_full),  caption="Full",        use_container_width=True)
    if up_black:
        p2.image(Image.open(up_black), caption="Black patch", use_container_width=True)
    if up_white:
        p3.image(Image.open(up_white), caption="White patch", use_container_width=True)

if up_full and up_black and up_white:
    if st.button("Run Inference"):
        up_full.seek(0)
        up_black.seek(0)
        up_white.seek(0)

        full_img  = Image.open(io.BytesIO(up_full.read()))
        black_img = Image.open(io.BytesIO(up_black.read()))
        white_img = Image.open(io.BytesIO(up_white.read()))

        model, device = load_model()

        with st.spinner("Running inference…"):
            pred_class, probs = run_inference(model, device, full_img, black_img, white_img)

        confidence = probs[pred_class] * 100

        bars_html = "".join(
            f"""<div class="bar-row">
                  <div class="bar-name">{CLASS_NAMES[i]}</div>
                  <div class="bar-bg">
                    <div class="bar-fill" style="width:{probs[i]*100:.1f}%"></div>
                  </div>
                  <div class="bar-pct">{probs[i]*100:.1f}%</div>
                </div>"""
            for i in range(NUM_CLASSES)
        )

        st.markdown(
            f"""
            <div class="result-card">
              <div class="sublabel">PREDICTED CLASS</div>
              <div class="label">{CLASS_NAMES[pred_class]}</div>
              <div class="sublabel" style="margin-top:0.5rem;">
                Confidence: <strong>{confidence:.1f}%</strong>
              </div>
              <div class="bar-wrap">{bars_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("⬆ Upload all three images to enable inference.")

st.divider()
st.caption("Javeria Rahman (i230137) | Shireen Fatima (i230130) | BS-AI")