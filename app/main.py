"""
Exam Script Evaluator — FastAPI 0.162 Backend Monolith (Module 1 Service).
Aligned with SRS Section 4.3 & 4.4 (ExternalInterfaces.docx):
- Python 3.12+
- FastAPI 0.162.0
- Pydantic v2 (extra="forbid")
- LangGraph 1.2.0
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.module1 import McqExtractionRequest, McqExtractionResponse
from app.services.mcq_extractor import extract_mcq_from_images

# Load environment variables from backend/.env or frontend/.env
backend_env = Path(__file__).resolve().parent.parent / ".env"
frontend_env = Path(__file__).resolve().parent.parent.parent / "frontend" / ".env"
if backend_env.exists():
    load_dotenv(backend_env)
elif frontend_env.exists():
    load_dotenv(frontend_env)

app = FastAPI(
    title="Exam Script Evaluator API",
    version="1.0.0",
    description="Module 1 (Script Intake and Answer Evidence) FastAPI 0.162 Service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "exam-script-evaluator-backend",
        "fastapi_version": "0.162.0",
        "module": "Module 1 - Script Intake and Answer Evidence",
    }


@app.post("/api/v1/module1/ocr/mcq", response_model=McqExtractionResponse)
async def extract_mcq_endpoint(payload: McqExtractionRequest) -> McqExtractionResponse:
    model_used, items = await extract_mcq_from_images(payload.images)
    return McqExtractionResponse(
        ok=True,
        model=model_used,
        items=items,
    )

