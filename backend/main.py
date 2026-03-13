"""
FastAPI backend server cho Design Check AI System.
"""
import os
import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from agents.design_check_agent import DesignCheckAgent

# -----------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------
app = FastAPI(
    title="Design Check AI",
    description="He thong AI kiem tra loi thiet ke 2D bang RAG + Qwen VL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# -----------------------------------------------------------------------
# Lazy-load agent
# -----------------------------------------------------------------------
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        _agent = DesignCheckAgent(api_key=api_key)
    return _agent


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------
@app.get("/")
async def serve_frontend():
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {"message": "Design Check AI backend is running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Design Check AI"}


@app.post("/analyze")
async def analyze_design(
    file: UploadFile = File(...),
    query: str = Form(default="graphic design poster advertisement"),
):
    """
    Nhan anh thiet ke va tra ve danh sach loi + bounding boxes.
    """
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    content_type = file.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Chi ho tro JPEG, PNG, WEBP."
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File rong.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File qua lon (max 10MB).")

    try:
        agent = get_agent()
        result = agent.analyze(
            image_bytes=image_bytes,
            filename=file.filename or "image.jpg",
            query=query,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
