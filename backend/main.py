from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.deepfake_detector import DeepfakeDetector
import shutil
from pathlib import Path
import uuid
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("deepguard")

app = FastAPI(
    title="DeepGuard AI API",
    version="2.0.0",
    description="Professional deepfake detection API with TTA and forensic analysis",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup
detector = DeepfakeDetector()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MIN_FILE_SIZE = 1024  # 1 KB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


@app.get("/")
def root():
    return {
        "message": "DeepGuard AI is running 🛡️",
        "status": "active",
        "version": "2.0.0",
        "capabilities": [
            "Test-Time Augmentation (TTA)",
            "Multi-scale face analysis",
            "Pixel-level forensic metrics",
            "Confidence calibration",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
        "device": str(detector.device),
        "label_mapping": detector.id2label,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    request_start = time.time()

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. Accepted: JPG, PNG, WebP",
        )

    # Read file content to check size
    content = await file.read()
    file_size = len(content)

    if file_size < MIN_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too small — must be at least 1 KB")
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large — maximum 20 MB")

    # Save file temporarily
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    try:
        with file_path.open("wb") as buffer:
            buffer.write(content)

        logger.info(f"Processing: {file.filename} ({file_size / 1024:.1f} KB)")

        result = detector.predict(str(file_path))
        file_path.unlink()  # Delete after processing

        total_time = round((time.time() - request_start) * 1000, 1)
        logger.info(
            f"Result: {result['prediction']} ({result['confidence']}%) in {total_time}ms"
        )

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "file_size_kb": round(file_size / 1024, 1),
            "result": result,
            "total_request_time_ms": total_time,
        })

    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
