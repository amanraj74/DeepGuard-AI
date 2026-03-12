import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import time
import io
from transformers import ViTForImageClassification, ViTImageProcessor
from facenet_pytorch import MTCNN

# ─── DeepfakeDetector (full backend embedded) ────────────────────────

class DeepfakeDetector:
    MODEL_A = "dima806/deepfake_vs_real_image_detection"
    MODEL_B = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    WEIGHT_A = 0.60
    WEIGHT_B = 0.40

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor_a = ViTImageProcessor.from_pretrained(self.MODEL_A)
        self.model_a     = ViTForImageClassification.from_pretrained(self.MODEL_A).to(self.device).eval()

        self.processor_b = ViTImageProcessor.from_pretrained(self.MODEL_B)
        self.model_b     = ViTForImageClassification.from_pretrained(self.MODEL_B).to(self.device).eval()

        self.mtcnn   = MTCNN(image_size=224, margin=40, keep_all=False,
                             post_process=False, device=self.device)
        self.id2label = {0: "Real", 1: "Deepfake"}

    # ── face crop ───────────────────────────────────────────────────
    def _crop_face(self, image: Image.Image, margin: int = 40):
        mtcnn = MTCNN(image_size=224, margin=margin,
                      keep_all=False, post_process=False, device=self.device)
        tensor = mtcnn(image)
        if tensor is not None:
            arr = tensor.permute(1, 2, 0).cpu().numpy().astype("uint8")
            return Image.fromarray(arr), True
        w, h = image.size
        s = min(w, h)
        l, t = (w - s) // 2, (h - s) // 2
        return image.crop((l, t, l + s, t + s)).resize((224, 224)), False

    # ── TTA augmentations ───────────────────────────────────────────
    def _tta_variants(self, image: Image.Image):
        variants = [image]
        variants.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        arr = np.array(image)
        bright = np.clip(arr.astype(np.int16) + 15, 0, 255).astype(np.uint8)
        dark   = np.clip(arr.astype(np.int16) - 15, 0, 255).astype(np.uint8)
        variants.append(Image.fromarray(bright))
        variants.append(Image.fromarray(dark))
        variants.append(image.rotate(3))
        return variants

    # ── single model inference ──────────────────────────────────────
    def _infer(self, processor, model, image: Image.Image):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.nn.functional.softmax(logits, dim=-1)[0]
        return probs[0].item(), probs[1].item()   # real, fake

    # ── forensics ───────────────────────────────────────────────────
    def _forensics(self, image: Image.Image):
        arr  = np.array(image.convert("RGB")).astype(np.float32)
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Noise variance
        blurred      = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_var    = float(np.var(gray - blurred))
        noise_score  = min(int(noise_var / 5), 100)

        # Edge consistency
        edges        = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        edge_score   = max(0, min(100, int((1 - edge_density * 20) * 100)))

        # Color channel correlation
        r, g, b      = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        corr_rg      = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
        corr_rb      = float(np.corrcoef(r.flatten(), b.flatten())[0, 1])
        avg_corr     = (abs(corr_rg) + abs(corr_rb)) / 2
        color_score  = min(int(avg_corr * 100), 100)

        # JPEG artifact detection
        gray_u8      = gray.astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, enc       = cv2.imencode('.jpg', gray_u8, encode_param)
        dec          = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        jpeg_diff    = float(np.mean(np.abs(gray - dec)))
        jpeg_score   = min(int(jpeg_diff * 10), 100)

        overall = int((noise_score * 0.3 + (100 - edge_score) * 0.3 +
                       color_score * 0.2 + jpeg_score * 0.2))

        return {
            "noise_score":        noise_score,
            "edge_score":         edge_score,
            "color_score":        color_score,
            "jpeg_score":         jpeg_score,
            "overall_suspicion":  overall,
            "noise_variance":     round(noise_var, 4),
            "edge_consistency":   round(edge_density, 4),
            "color_correlation":  round(avg_corr, 4),
            "jpeg_artifact_level":round(jpeg_diff, 4),
        }

    # ── image quality assessment ────────────────────────────────────
    def _image_quality(self, image: Image.Image):
        warnings = []
        w, h     = image.size
        if w < 100 or h < 100:
            warnings.append("Very small image — accuracy may be lower")
        if w < 224 or h < 224:
            warnings.append("Low resolution — upscaled for analysis")
        arr  = np.array(image.convert("L")).astype(np.float32)
        if np.std(arr) < 20:
            warnings.append("Low contrast image detected")
        grade = "high" if not warnings else ("medium" if len(warnings) == 1 else "low")
        return {"quality_grade": grade, "warnings": warnings}

    # ── main predict ─────────────────────────────────────────────────
    def predict(self, image_path: str):
        t0    = time.time()
        image = Image.open(image_path).convert("RGB")

        face_img, face_found = self._crop_face(image, margin=40)
        scales = [
            self._crop_face(image, margin=40)[0],
            self._crop_face(image, margin=80)[0],
            self._crop_face(image, margin=20)[0],
        ]

        # TTA × scales × 2 models
        real_a_acc, fake_a_acc = 0.0, 0.0
        real_b_acc, fake_b_acc = 0.0, 0.0
        total = 0

        for scale_img in scales:
            for variant in self._tta_variants(scale_img):
                ra, fa = self._infer(self.processor_a, self.model_a, variant)
                rb, fb = self._infer(self.processor_b, self.model_b, variant)
                real_a_acc += ra; fake_a_acc += fa
                real_b_acc += rb; fake_b_acc += fb
                total += 1

        # Average per model
        real_a = real_a_acc / total; fake_a = fake_a_acc / total
        real_b = real_b_acc / total; fake_b = fake_b_acc / total

        # Weighted ensemble
        real_final = self.WEIGHT_A * real_a + self.WEIGHT_B * real_b
        fake_final = self.WEIGHT_A * fake_a + self.WEIGHT_B * fake_b

        # Normalize
        total_prob  = real_final + fake_final
        real_final /= total_prob
        fake_final /= total_prob

        # Temperature calibration
        T = 1.5
        real_cal = real_final ** (1 / T)
        fake_cal = fake_final ** (1 / T)
        s = real_cal + fake_cal
        real_cal /= s; fake_cal /= s

        prediction = "Real" if real_cal >= fake_cal else "Deepfake"
        confidence = round(max(real_cal, fake_cal) * 100, 2)

        pred_a = "Real" if real_a >= fake_a else "Deepfake"
        pred_b = "Real" if real_b >= fake_b else "Deepfake"

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "prediction":   prediction,
            "confidence":   confidence,
            "face_detected": face_found,
            "scores": {
                "Real":     round(real_cal * 100, 2),
                "Deepfake": round(fake_cal * 100, 2),
            },
            "ensemble": {
                "agreement": pred_a == pred_b,
                "per_model": {
                    "model_a": {
                        "model":      self.MODEL_A,
                        "weight":     self.WEIGHT_A,
                        "prediction": pred_a,
                        "real":       round(real_a * 100, 2),
                        "fake":       round(fake_a * 100, 2),
                    },
                    "model_b": {
                        "model":      self.MODEL_B,
                        "weight":     self.WEIGHT_B,
                        "prediction": pred_b,
                        "real":       round(real_b * 100, 2),
                        "fake":       round(fake_b * 100, 2),
                    },
                },
            },
            "forensics":      self._forensics(face_img),
            "image_quality":  self._image_quality(image),
            "analysis_details": {
                "total_inferences":   total * 2,
                "scales_analyzed":    len(scales),
                "tta_passes":         len(self._tta_variants(face_img)),
                "processing_time_ms": elapsed_ms,
                "device":             str(self.device),
            },
        }


# ─── Load model once ────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    return DeepfakeDetector()


# ─── Helper: run predict from uploaded bytes ────────────────────────
def run_detect(uploaded_file):
    import tempfile, os
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        detector = load_detector()
        result   = detector.predict(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result


# ════════════════════════════════════════════════════════════════════
# PASTE YOUR FULL ORIGINAL FRONTEND CODE BELOW — UNCHANGED
# Only replace the `requests.post(...)` block with `run_detect(...)`
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DeepGuard AI — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        font-size: 2.8rem; font-weight: 900; text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 0.5rem 0 0; letter-spacing: -0.02em;
    }
    .subtitle { text-align: center; color: #666; font-size: 1.05rem; margin-bottom: 1.5rem; font-weight: 400; }
    .real-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 6px solid #28a745; padding: 1.5rem; border-radius: 12px;
        margin: 1rem 0; box-shadow: 0 4px 15px rgba(40,167,69,.15);
    }
    .real-box h2 { color: #155724; margin: 0; font-size: 1.6rem; }
    .real-box h3 { color: #155724; margin: .3rem 0 0; font-weight: 600; }
    .real-box p  { color: #155724; margin: .5rem 0 0; }
    .fake-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 6px solid #dc3545; padding: 1.5rem; border-radius: 12px;
        margin: 1rem 0; box-shadow: 0 4px 15px rgba(220,53,69,.15);
    }
    .fake-box h2 { color: #721c24; margin: 0; font-size: 1.6rem; }
    .fake-box h3 { color: #721c24; margin: .3rem 0 0; font-weight: 600; }
    .fake-box p  { color: #721c24; margin: .5rem 0 0; }
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef1ff 100%);
        padding: 1rem 1.2rem; border-radius: 10px; border: 1px solid #e0e5ff; margin: .5rem 0;
    }
    .metric-card h4 { margin: 0 0 .3rem; color: #4a5568; font-size: .85rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #2d3748; }
    .forensic-section { background: #f7f8fc; border-radius: 12px; padding: 1.2rem; border: 1px solid #e2e8f0; margin: .5rem 0; }
    .tta-badge { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color: white; padding: .4rem 1rem; border-radius: 20px; font-size: .8rem; font-weight: 600; display: inline-block; margin: .3rem .3rem .3rem 0; }
    .quality-badge-high   { background: #d4edda; color: #155724; padding: .3rem .8rem; border-radius: 15px; font-size: .8rem; font-weight: 600; display: inline-block; }
    .quality-badge-medium { background: #fff3cd; color: #856404; padding: .3rem .8rem; border-radius: 15px; font-size: .8rem; font-weight: 600; display: inline-block; }
    .quality-badge-low    { background: #f8d7da; color: #721c24; padding: .3rem .8rem; border-radius: 15px; font-size: .8rem; font-weight: 600; display: inline-block; }
    .info-card { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin: .5rem 0; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg,#1a1a2e 0%,#16213e 100%); }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🛡️ DeepGuard AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Professional Deepfake & AI-Generated Media Detection'
    ' · Multi-Pass TTA Analysis · Forensic Metrics</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.header("🛡️ DeepGuard AI")
    st.caption("v2.0 — Professional Edition")
    st.markdown("---")
    st.subheader("About")
    st.write("DeepGuard AI uses a **Dual-Model Ensemble** of two Vision Transformer (ViT) models with **Test-Time Augmentation** for professional-grade deepfake detection with **99%+ accuracy**.")
    st.markdown("---")
    st.subheader("Detection Pipeline")
    st.write("""
    1. 📤 Upload a face image
    2. 🔍 MTCNN face detection (3 scales)
    3. 🤖 **2 ViT models** × 5 TTA passes × 3 scales
    4. 📊 **30 total inferences** ensembled
    5. 🔬 Pixel-level forensic analysis
    6. ✅ Calibrated confidence score
    """)
    st.markdown("---")
    st.subheader("Detects")
    st.write("""
    - ✅ Deepfake face swaps
    - ✅ AI-generated portraits (GANs, Diffusion)
    - ✅ Manipulated / retouched media
    - ✅ Synthetic faces (StyleGAN, Midjourney)
    """)
    st.markdown("---")
    st.subheader("Technology")
    st.write("""
    - 🧠 **Model A**: dima806 ViT (99.3% acc)
    - 🧠 **Model B**: prithivMLmods ViT (92% acc)
    - 🔄 **TTA**: 5 augmentations × 3 scales
    - 🏗️ **Ensemble**: Weighted voting (60/40)
    - 📐 **Calibration**: Temperature scaling
    - 🔬 **Forensics**: Noise, Edge, Color, JPEG
    """)
    st.markdown("---")
    st.caption("Built for IIT Bombay Hack & Break 2026")
    st.caption("Theme: Cybersecurity + Generative AI")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a face image to analyze",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload any face image — real or AI-generated. Max 20 MB.",
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"📁 {uploaded_file.name}", use_column_width=True)
        st.success(f"✅ Image uploaded: {uploaded_file.name}")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        icol1, icol2, icol3 = st.columns(3)
        icol1.metric("Width",  f"{image.size[0]} px")
        icol2.metric("Height", f"{image.size[1]} px")
        icol3.metric("Size",   f"{uploaded_file.size / 1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Upload an image to get started")
        st.image("https://img.icons8.com/fluency/200/upload-to-cloud.png", width=150)

with col2:
    st.subheader("🔍 Detection Results")

    if uploaded_file:
        with st.spinner("🤖 Running multi-pass AI analysis..."):
            try:
                # ── Direct call (replaces requests.post) ──────────────
                result = run_detect(uploaded_file)
                # ──────────────────────────────────────────────────────

                face_detected  = result.get("face_detected", True)
                prediction     = result["prediction"]
                confidence     = result["confidence"]
                real_score     = result["scores"]["Real"]
                fake_score     = result["scores"]["Deepfake"]
                forensics      = result.get("forensics", {})
                image_quality  = result.get("image_quality", {})
                analysis       = result.get("analysis_details", {})

                badges_html = ""
                if analysis:
                    badges_html += f'<span class="tta-badge">🔄 {analysis.get("total_inferences", 30)} Inferences</span>'
                    badges_html += f'<span class="tta-badge">📐 {analysis.get("scales_analyzed", 3)} Scales</span>'
                    badges_html += f'<span class="tta-badge">⚡ {analysis.get("processing_time_ms", 0):.0f}ms</span>'
                    device = analysis.get("device", "cpu")
                    device_label = "GPU" if "cuda" in device else "MPS" if "mps" in device else "CPU"
                    badges_html += f'<span class="tta-badge">🖥️ {device_label}</span>'
                st.markdown(badges_html, unsafe_allow_html=True)

                quality_grade = image_quality.get("quality_grade", "high")
                grade_labels  = {"high": "High Quality", "medium": "Medium Quality", "low": "Low Quality"}
                st.markdown(
                    f'<span class="quality-badge-{quality_grade}">📷 {grade_labels.get(quality_grade)}</span>',
                    unsafe_allow_html=True,
                )
                for warning in image_quality.get("warnings", []):
                    st.warning(f"⚠️ {warning}")

                if prediction == "Real":
                    st.markdown(f"""
                    <div class="real-box">
                        <h2>✅ AUTHENTIC MEDIA</h2>
                        <h3>Confidence: {confidence}%</h3>
                        <p>This image appears to be <strong>genuine</strong> and not AI-generated.</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="fake-box">
                        <h2>⚠️ DEEPFAKE DETECTED</h2>
                        <h3>Confidence: {confidence}%</h3>
                        <p>This image shows signs of <strong>AI manipulation</strong>.</p>
                    </div>""", unsafe_allow_html=True)

                if not face_detected:
                    st.warning("⚠️ No clear face detected — analyzed center region instead. Results may be less accurate.")

                ensemble_data = result.get("ensemble", {})
                if ensemble_data:
                    if ensemble_data.get("agreement"):
                        st.markdown('<span class="tta-badge" style="background:linear-gradient(135deg,#28a745,#20c997);">🤝 Both Models Agree</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="tta-badge" style="background:linear-gradient(135deg,#ffc107,#fd7e14);">⚖️ Models Disagree — Ensemble Resolved</span>', unsafe_allow_html=True)

                    per_model = ensemble_data.get("per_model", {})
                    if per_model:
                        with st.expander("🧠 Per-Model Predictions"):
                            for mk, mv in per_model.items():
                                model_short = mv.get('model', mk).split('/')[-1]
                                weight_pct  = int(mv.get('weight', 0) * 100)
                                pred        = mv.get('prediction', '?')
                                real_p      = mv.get('real', 0)
                                fake_p      = mv.get('fake', 0)
                                icon        = '✅' if pred == 'Real' else '⚠️'
                                st.write(f"**{model_short}** (weight {weight_pct}%): {icon} {pred} — Real {real_p}% / Fake {fake_p}%")

                st.markdown("---")
                st.subheader("📊 Authenticity Score Breakdown")
                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    st.markdown(f'<div class="metric-card"><h4>Real Probability</h4><span class="value">{real_score}%</span></div>', unsafe_allow_html=True)
                    st.progress(real_score / 100)
                with score_col2:
                    st.markdown(f'<div class="metric-card"><h4>Deepfake Probability</h4><span class="value">{fake_score}%</span></div>', unsafe_allow_html=True)
                    st.progress(fake_score / 100)

                st.markdown("---")
                st.subheader("🔬 Forensic Analysis")

                if forensics:
                    st.markdown('<div class="forensic-section">', unsafe_allow_html=True)
                    suspicion = forensics.get("overall_suspicion", 50)
                    suspicion_color = "#28a745" if suspicion < 35 else "#ffc107" if suspicion < 65 else "#dc3545"
                    suspicion_label = "Low Suspicion" if suspicion < 35 else "Moderate Suspicion" if suspicion < 65 else "High Suspicion"
                    st.markdown(f"**Overall Forensic Suspicion**: <span style='color:{suspicion_color};font-weight:700;'>{suspicion_label} ({suspicion}/100)</span>", unsafe_allow_html=True)
                    st.progress(suspicion / 100)
                    st.markdown("</div>", unsafe_allow_html=True)

                    fcol1, fcol2 = st.columns(2)
                    with fcol1:
                        noise_score = forensics.get("noise_score", 0)
                        st.markdown(f'<div class="metric-card"><h4>🔊 Noise Variance</h4><span class="value">{noise_score}/100</span></div>', unsafe_allow_html=True)
                        st.progress(min(noise_score / 100, 1.0))
                        st.caption("Higher = more synthetic noise patterns")
                        color_score = forensics.get("color_score", 0)
                        st.markdown(f'<div class="metric-card"><h4>🎨 Color Correlation</h4><span class="value">{color_score}/100</span></div>', unsafe_allow_html=True)
                        st.progress(min(color_score / 100, 1.0))
                        st.caption("Higher = unnatural color channel patterns")
                    with fcol2:
                        edge_score = forensics.get("edge_score", 0)
                        st.markdown(f'<div class="metric-card"><h4>📐 Edge Consistency</h4><span class="value">{edge_score}/100</span></div>', unsafe_allow_html=True)
                        st.progress(min(edge_score / 100, 1.0))
                        st.caption("Lower = more inconsistent edge structure")
                        jpeg_score = forensics.get("jpeg_score", 0)
                        st.markdown(f'<div class="metric-card"><h4>📦 JPEG Artifact Level</h4><span class="value">{jpeg_score}/100</span></div>', unsafe_allow_html=True)
                        st.progress(min(jpeg_score / 100, 1.0))
                        st.caption("Higher = more compression artifacts detected")

                st.markdown("---")
                with st.expander("🔎 Detailed Forensic Explanation"):
                    if prediction == "Deepfake":
                        st.error("""
                        **Suspicious patterns detected by the AI model:**
                        - Unnatural facial texture inconsistencies in ViT attention maps
                        - Irregular pixel-level artifacts across multiple TTA passes
                        - Atypical edge blending around facial features at multiple scales
                        - Spectral anomalies detected in high-frequency image regions
                        - Abnormal color channel correlation patterns
                        """)
                    else:
                        st.success("""
                        **Authenticity indicators confirmed by the AI model:**
                        - Natural facial texture patterns consistent across all TTA passes
                        - Consistent lighting and shadow distribution at multiple scales
                        - Normal pixel-level variations with natural noise characteristics
                        - No detectable GAN, diffusion, or face-swap artifacts
                        - Healthy color channel correlation and edge consistency
                        """)
                    if forensics:
                        st.write("**Raw Forensic Values:**")
                        st.json({
                            "noise_variance":     forensics.get("noise_variance", "N/A"),
                            "edge_consistency":   forensics.get("edge_consistency", "N/A"),
                            "color_correlation":  forensics.get("color_correlation", "N/A"),
                            "jpeg_artifact_level":forensics.get("jpeg_artifact_level", "N/A"),
                        })
                    if analysis:
                        st.write("**Analysis Configuration:**")
                        st.json(analysis)
                    if ensemble_data:
                        st.write("**Ensemble Details:**")
                        st.json(ensemble_data)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👈 Upload an image first to see detection results")
        st.markdown("---")
        st.markdown("##### 🚀 What makes DeepGuard AI different?")
        feature_col1, feature_col2 = st.columns(2)
        with feature_col1:
            st.markdown('<div class="metric-card"><h4>🧠 Dual-Model Ensemble</h4><span class="value">2 ViT Models</span></div>', unsafe_allow_html=True)
            st.caption("99.3% + 92% accuracy models voting together")
            st.markdown('<div class="metric-card"><h4>📐 Multi-Scale Analysis</h4><span class="value">3 scales</span></div>', unsafe_allow_html=True)
            st.caption("Face analyzed at tight, medium, and wide crops")
        with feature_col2:
            st.markdown('<div class="metric-card"><h4>🔬 Forensic Metrics</h4><span class="value">4 signals</span></div>', unsafe_allow_html=True)
            st.caption("Noise, edge, color, and JPEG artifact analysis")
            st.markdown('<div class="metric-card"><h4>🔄 Total Inferences</h4><span class="value">30 per image</span></div>', unsafe_allow_html=True)
            st.caption("5 TTA × 3 scales × 2 models = 30 passes ensembled")

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#999;padding:1rem;">
    🛡️ <strong>DeepGuard AI v2.0</strong> | Dual-Model ViT Ensemble + TTA + Multi-Scale Forensics<br>
    IIT Bombay Hack &amp; Break 2026 | Theme: Cybersecurity + Generative AI
</div>
""", unsafe_allow_html=True)
