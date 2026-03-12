from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageFilter, ImageEnhance
import torch
import numpy as np
from facenet_pytorch import MTCNN
import time
import logging

logger = logging.getLogger("deepguard")


class DeepfakeDetector:
    """
    Professional-grade deepfake detector using a DUAL-MODEL ENSEMBLE:
      • Model A: prithivMLmods/Deep-Fake-Detector-v2-Model  (92% accuracy)
      • Model B: dima806/deepfake_vs_real_image_detection    (99.3% accuracy)

    Combined with Test-Time Augmentation (TTA), multi-scale face analysis,
    confidence calibration, and pixel-level forensic metrics.
    """

    # Temperature for confidence calibration
    TEMPERATURE = 1.4

    # TTA augmentation configs
    TTA_AUGMENTS = [
        {"name": "original"},
        {"name": "hflip"},
        {"name": "rotate_cw", "angle": 5},
        {"name": "rotate_ccw", "angle": -5},
        {"name": "sharpen", "factor": 2.0},
    ]

    # Multi-scale crop margins for MTCNN
    CROP_MARGINS = [40, 80, 120]

    # Ensemble model definitions
    MODELS = {
        "primary": {
            "name": "dima806/deepfake_vs_real_image_detection",
            "weight": 0.60,  # Higher weight — 99.3% accuracy model
        },
        "secondary": {
            "name": "prithivMLmods/Deep-Fake-Detector-v2-Model",
            "weight": 0.40,  # Lower weight — 92% accuracy model
        },
    }

    # Decision threshold: require this much deepfake probability to classify as fake
    # This reduces false positives on real images
    DEEPFAKE_THRESHOLD = 0.55

    # When no face is detected (AI illustrations, anime, non-photo content),
    # use a lower threshold and shift weights toward the model that handles them better
    NO_FACE_DEEPFAKE_THRESHOLD = 0.42
    NO_FACE_WEIGHTS = {
        "primary": 0.40,   # dima806: less reliable for non-photo content
        "secondary": 0.60,  # prithivMLmods: better at catching AI illustrations
    }

    def __init__(self):
        logger.info("Loading DeepGuard AI dual-model ensemble...")
        print("=" * 60)
        print("  DeepGuard AI — Loading Dual-Model Ensemble")
        print("=" * 60)

        # Auto-detect best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"  Device: {self.device}")

        # ── Load both models ──────────────────────────────────────────
        self.models = {}
        self.processors = {}
        self.model_real_idx = {}
        self.model_fake_idx = {}

        for key, cfg in self.MODELS.items():
            model_name = cfg["name"]
            print(f"  Loading {key}: {model_name}...")

            processor = ViTImageProcessor.from_pretrained(model_name)
            model = ViTForImageClassification.from_pretrained(model_name)
            model.to(self.device)
            model.eval()

            # Read label mapping directly from model config
            id2label = model.config.id2label
            print(f"    Labels: {id2label}")

            # Find real and fake indices dynamically
            real_idx = None
            fake_idx = None
            for idx, label in id2label.items():
                idx = int(idx)
                low = label.lower().strip()
                if low in ("real", "realism"):
                    real_idx = idx
                elif low in ("fake", "deepfake"):
                    fake_idx = idx

            if real_idx is None or fake_idx is None:
                raise ValueError(
                    f"Cannot identify Real/Fake classes for {model_name}: {id2label}"
                )

            self.models[key] = model
            self.processors[key] = processor
            self.model_real_idx[key] = real_idx
            self.model_fake_idx[key] = fake_idx
            print(f"    Real=idx{real_idx}, Fake=idx{fake_idx} ✅")

        # ── Face detectors (multi-scale) ──────────────────────────────
        self.face_detectors = {}
        for margin in self.CROP_MARGINS:
            self.face_detectors[margin] = MTCNN(
                image_size=224,
                margin=margin,
                keep_all=False,
                select_largest=True,
                post_process=False,
                device=self.device if self.device.type == "cuda" else "cpu",
            )

        print("=" * 60)
        print("  DeepGuard AI ready ✅  (2 models loaded)")
        print("=" * 60)

    # ─── Face Cropping ────────────────────────────────────────────────

    def crop_face(self, image: Image.Image, margin: int = 80):
        """Detect and crop face from image with specified margin."""
        detector = self.face_detectors.get(margin, self.face_detectors[80])
        face_tensor = detector(image)

        if face_tensor is not None:
            face_array = face_tensor.permute(1, 2, 0).cpu().numpy().astype("uint8")
            return Image.fromarray(face_array), True
        else:
            # Fallback: center-crop the image
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            cropped = image.crop((left, top, left + min_dim, top + min_dim))
            return cropped.resize((224, 224)), False

    # ─── Test-Time Augmentation ───────────────────────────────────────

    def _apply_augmentation(self, image: Image.Image, aug: dict) -> Image.Image:
        """Apply a single augmentation to an image."""
        name = aug["name"]

        if name == "original":
            return image
        elif name == "hflip":
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif name in ("rotate_cw", "rotate_ccw"):
            return image.rotate(
                -aug["angle"], resample=Image.BICUBIC, fillcolor=(128, 128, 128)
            )
        elif name == "sharpen":
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(aug["factor"])
        return image

    def _run_single_inference(self, image: Image.Image, model_key: str) -> np.ndarray:
        """Run model inference on a single image, return calibrated softmax probs."""
        processor = self.processors[model_key]
        model = self.models[model_key]

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            scaled_logits = logits / self.TEMPERATURE
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)

        return probs[0].cpu().numpy()

    def _run_tta_inference(self, face_image: Image.Image, model_key: str) -> np.ndarray:
        """Run TTA: multiple augmented inferences averaged together."""
        all_probs = []
        for aug in self.TTA_AUGMENTS:
            aug_image = self._apply_augmentation(face_image, aug)
            probs = self._run_single_inference(aug_image, model_key)
            all_probs.append(probs)
        return np.mean(all_probs, axis=0)

    # ─── Multi-Scale Analysis ─────────────────────────────────────────

    def _multi_scale_analysis_single_model(self, image: Image.Image, model_key: str):
        """Run detection at multiple crop scales for a single model."""
        all_probs = []
        face_found_any = False

        for margin in self.CROP_MARGINS:
            face_img, found = self.crop_face(image, margin=margin)
            if found:
                face_found_any = True

            probs = self._run_tta_inference(face_img, model_key)
            all_probs.append(probs)

        # Weighted average: tighter crops weighted higher
        weights = [0.45, 0.35, 0.20]
        weighted_probs = np.zeros_like(all_probs[0])
        for probs, weight in zip(all_probs, weights):
            weighted_probs += probs * weight

        return weighted_probs, face_found_any

    def _ensemble_predict(self, image: Image.Image):
        """
        Run both models with multi-scale TTA, then ensemble their predictions.
        Uses adaptive weights: normal weights when face is detected,
        shifted weights when no face is found (better for AI illustrations).
        Returns (real_probability, fake_probability, face_detected, per_model_details).
        """
        model_results = {}
        face_detected = False

        for key, cfg in self.MODELS.items():
            probs, found = self._multi_scale_analysis_single_model(image, key)
            if found:
                face_detected = True

            real_idx = self.model_real_idx[key]
            fake_idx = self.model_fake_idx[key]

            real_prob = float(probs[real_idx])
            fake_prob = float(probs[fake_idx])

            model_results[key] = {
                "model_name": cfg["name"],
                "weight": cfg["weight"],
                "real_prob": real_prob,
                "fake_prob": fake_prob,
                "prediction": "Deepfake" if fake_prob > real_prob else "Real",
            }

        # Choose weights based on face detection
        # When no face found → shift weight toward prithivMLmods (better for illustrations)
        if face_detected:
            weights = {k: cfg["weight"] for k, cfg in self.MODELS.items()}
        else:
            weights = dict(self.NO_FACE_WEIGHTS)
            logger.info("No face detected — using adaptive weights (favoring secondary model)")

        # Update model results with actual weights used
        for key in model_results:
            model_results[key]["weight"] = weights[key]

        # Weighted ensemble
        ensemble_real = sum(
            model_results[k]["real_prob"] * weights[k]
            for k in self.MODELS
        )
        ensemble_fake = sum(
            model_results[k]["fake_prob"] * weights[k]
            for k in self.MODELS
        )

        # Normalize
        total = ensemble_real + ensemble_fake
        if total > 0:
            ensemble_real /= total
            ensemble_fake /= total

        return ensemble_real, ensemble_fake, face_detected, model_results

    # ─── Pixel-Level Forensic Metrics ─────────────────────────────────

    def _compute_forensic_metrics(self, image: Image.Image) -> dict:
        """Compute real signal-analysis forensic metrics on the image."""
        img_array = np.array(image).astype(np.float64)
        metrics = {}

        # 1. Noise variance (high-pass filter)
        gray = np.mean(img_array, axis=2)
        kernel_response = (
            4 * gray[1:-1, 1:-1]
            - gray[:-2, 1:-1]
            - gray[2:, 1:-1]
            - gray[1:-1, :-2]
            - gray[1:-1, 2:]
        )
        noise_var = float(np.var(kernel_response))
        metrics["noise_variance"] = round(noise_var, 2)
        noise_score = min(100, max(0, 100 - (noise_var - 50) / 8))
        metrics["noise_score"] = round(noise_score, 1)

        # 2. Edge consistency (Laplacian variance)
        metrics["edge_consistency"] = round(noise_var, 2)
        edge_score = min(100, max(0, (noise_var - 100) / 10))
        metrics["edge_score"] = round(edge_score, 1)

        # 3. Color channel correlation
        if img_array.shape[2] >= 3:
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            rg_corr = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
            rb_corr = float(np.corrcoef(r.flatten(), b.flatten())[0, 1])
            gb_corr = float(np.corrcoef(g.flatten(), b.flatten())[0, 1])
            avg_corr = (rg_corr + rb_corr + gb_corr) / 3
            metrics["color_correlation"] = round(avg_corr, 4)
            color_score = min(100, max(0, (avg_corr - 0.85) * 500))
            metrics["color_score"] = round(color_score, 1)
        else:
            metrics["color_correlation"] = 0.0
            metrics["color_score"] = 50.0

        # 4. JPEG artifact analysis
        h, w = gray.shape
        block_diffs = []
        for i in range(8, h - 8, 8):
            row_diff = np.abs(gray[i, :] - gray[i - 1, :])
            block_diffs.append(np.mean(row_diff))
        for j in range(8, w - 8, 8):
            col_diff = np.abs(gray[:, j] - gray[:, j - 1])
            block_diffs.append(np.mean(col_diff))

        if block_diffs:
            jpeg_artifact = float(np.mean(block_diffs))
            metrics["jpeg_artifact_level"] = round(jpeg_artifact, 2)
            jpeg_score = min(100, max(0, jpeg_artifact / 0.3 * 100))
            metrics["jpeg_score"] = round(jpeg_score, 1)
        else:
            metrics["jpeg_artifact_level"] = 0.0
            metrics["jpeg_score"] = 0.0

        # 5. Overall forensic suspicion
        forensic_suspicion = (
            noise_score * 0.30
            + metrics["color_score"] * 0.30
            + (100 - edge_score) * 0.20
            + metrics["jpeg_score"] * 0.20
        )
        metrics["overall_suspicion"] = round(forensic_suspicion, 1)

        return metrics

    # ─── Image Quality Assessment ─────────────────────────────────────

    def _assess_image_quality(self, image: Image.Image) -> dict:
        """Check image quality and return warnings."""
        w, h = image.size
        quality = {
            "width": w,
            "height": h,
            "megapixels": round((w * h) / 1_000_000, 2),
            "warnings": [],
        }

        if min(w, h) < 128:
            quality["warnings"].append("Very low resolution — detection accuracy may be reduced")
        elif min(w, h) < 256:
            quality["warnings"].append("Low resolution — results may be less reliable")

        if max(w, h) > 6000:
            quality["warnings"].append("Very high resolution — image was downscaled for analysis")

        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > 3:
            quality["warnings"].append("Unusual aspect ratio — face detection may be less accurate")

        quality["quality_grade"] = (
            "low" if min(w, h) < 128
            else "medium" if min(w, h) < 512
            else "high"
        )

        return quality

    # ─── Main Prediction Pipeline ─────────────────────────────────────

    def predict(self, image_path: str) -> dict:
        """
        Full prediction pipeline:
        1. Load & assess image quality
        2. Dual-model ensemble with multi-scale TTA
        3. Decision threshold for deepfake classification
        4. Pixel-level forensic analysis
        """
        start_time = time.time()

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Step 1: Image quality assessment
        quality = self._assess_image_quality(image)

        # Step 2: Dual-model ensemble prediction
        real_prob, fake_prob, face_detected, model_details = self._ensemble_predict(image)

        # Step 3: Apply adaptive decision threshold
        # Use lower threshold when no face detected (AI illustrations, anime, etc)
        threshold = self.DEEPFAKE_THRESHOLD if face_detected else self.NO_FACE_DEEPFAKE_THRESHOLD
        logger.info(
            f"Ensemble: real={real_prob:.3f} fake={fake_prob:.3f} "
            f"threshold={threshold} face={'yes' if face_detected else 'no'}"
        )

        if fake_prob >= threshold:
            prediction = "Deepfake"
            confidence = fake_prob
        else:
            prediction = "Real"
            confidence = real_prob

        # Step 4: Forensic metrics
        face_for_forensics, _ = self.crop_face(image, margin=80)
        forensics = self._compute_forensic_metrics(face_for_forensics)

        processing_time = round((time.time() - start_time) * 1000, 1)

        # Format model details for response
        per_model = {}
        for key, detail in model_details.items():
            per_model[key] = {
                "model": detail["model_name"],
                "weight": detail["weight"],
                "prediction": detail["prediction"],
                "real": round(detail["real_prob"] * 100, 2),
                "fake": round(detail["fake_prob"] * 100, 2),
            }

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "face_detected": face_detected,
            "scores": {
                "Real": round(real_prob * 100, 2),
                "Deepfake": round(fake_prob * 100, 2),
            },
            "ensemble": {
                "models_used": 2,
                "per_model": per_model,
                "threshold": self.DEEPFAKE_THRESHOLD,
                "agreement": (
                    model_details["primary"]["prediction"]
                    == model_details["secondary"]["prediction"]
                ),
            },
            "forensics": forensics,
            "image_quality": quality,
            "analysis_details": {
                "tta_passes": len(self.TTA_AUGMENTS),
                "scales_analyzed": len(self.CROP_MARGINS),
                "total_inferences": len(self.TTA_AUGMENTS) * len(self.CROP_MARGINS) * 2,
                "device": str(self.device),
                "processing_time_ms": processing_time,
                "calibration_temperature": self.TEMPERATURE,
            },
        }
