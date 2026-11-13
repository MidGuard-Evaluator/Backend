import asyncio
import base64
import io
import json
from collections import defaultdict
from typing import Any

import httpx
from PIL import Image, ImageEnhance, ImageOps

from app.schemas.module1 import (
    ExtractedFlagSchema,
    ExtractedSpanSchema,
    StructuredQuestionResultSchema,
    StructuredRegionPayload,
)
from app.services.mcq_extractor import _resolve_gemini_api_key, _resolve_gemini_model

STRUCTURED_OCR_PROMPT = """You are an ultra-precise academic handwritten answer script transcriber for university examinations.
You are inspecting cropped image region(s) of a single student's handwritten response to ONE question.
(Multiple images, if present, are continuation parts of the SAME question in page order.)

CRITICAL RULES:
1. Transcribe the handwritten text line by line faithfully, preserving exact technical terminology, mathematical notation, equations, and numbers. Do NOT invent or complete sentences that are not written.
2. Output strictly valid JSON matching this schema:
{
  "blankDetected": false,
  "hasDiagramOrTable": false,
  "lines": [
    {
      "lineNumber": 1,
      "text": "exact transcribed line text",
      "confidence": 0.96,
      "unclearNote": ""
    }
  ]
}
3. If the cropped region is completely blank (no handwritten answer, only ruled lines or empty margin), set "blankDetected": true and "lines": [].
4. If the region contains a hand-drawn diagram, flowchart, circuit, graph, or complex table that cannot be faithfully represented as plain text lines, set "hasDiagramOrTable": true and still transcribe any legible labels or prose into "lines".
5. If any word or symbol is smudged, crossed out, or ambiguous, mark the unclear token with `[?]` inside "text", set lower "confidence" (e.g. 0.55 - 0.72), and describe the ambiguity briefly in "unclearNote".
6. Every confidence value MUST be a float between 0.0 and 1.0."""


def _enhance_region_for_ocr(region: StructuredRegionPayload) -> tuple[str, str]:
    """Applies mild autocontrast & sharpening so faint pencil/pen handwriting is crisp."""
    try:
        raw_bytes = base64.b64decode(region.base64Data)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pil_img = ImageOps.autocontrast(pil_img, cutoff=1)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.3)
        out = io.BytesIO()
        pil_img.save(out, format="PNG", optimize=True)
        return "image/png", base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        return region.mimeType or "image/jpeg", region.base64Data


async def _extract_single_question_ocr(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    question_id: str,
    q_regions: list[StructuredRegionPayload],
) -> tuple[str, StructuredQuestionResultSchema]:
    q_regions_sorted = sorted(q_regions, key=lambda r: r.pageNumber)
    q_num = q_regions_sorted[0].questionNumber
    q_prompt = q_regions_sorted[0].questionPrompt or ""

    context_hint = f"Question {q_num}"
    if q_prompt:
        context_hint += f": {q_prompt}"

    image_parts: list[dict[str, Any]] = []
    for reg in q_regions_sorted:
        mime, b64 = _enhance_region_for_ocr(reg)
        image_parts.append({"inlineData": {"mimeType": mime, "data": b64}})

    parts: list[dict[str, Any]] = [
        {"text": f"{STRUCTURED_OCR_PROMPT}\n\nQuestion context (for terminology disambiguation only, do NOT copy into answer unless written): {context_hint}"},
        *image_parts,
    ]

    candidate_models = [model]
    if model != "gemini-3-flash-preview":
        candidate_models.append("gemini-3-flash-preview")

    resp = None
    active_model = model
    for idx_m, cand in enumerate(candidate_models):
        active_model = cand
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cand}:generateContent?key={api_key}"
        gen_config: dict[str, Any] = {"temperature": 0.0, "topK": 1, "seed": 42}
        if cand.startswith("gemini-"):
            gen_config["responseMimeType"] = "application/json"

        attempts = 2 if idx_m == 0 else 1
        for attempt in range(attempts):
            resp = await client.post(
                url,
                json={"contents": [{"parts": parts}], "generationConfig": gen_config},
            )
            if resp.status_code != 503:
                break
            if attempt < attempts - 1:
                await asyncio.sleep(1.5)

        if resp is not None and resp.status_code == 200:
            break
        if resp is None or resp.status_code != 503:
            break

    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "NO_RESPONSE"
        detail = resp.text[:240] if resp is not None else "No response"
        raise RuntimeError(f"Vision OCR HTTP {code}: {detail}")

    data = resp.json()
    raw_json_text = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    if not raw_json_text:
        raise RuntimeError("Empty response returned by Vision OCR model")

    cleaned = raw_json_text.strip().removeprefix("```json").removesuffix("```").strip()
    parsed = json.loads(cleaned)

    blank_detected = bool(parsed.get("blankDetected", False))
    has_diagram = bool(parsed.get("hasDiagramOrTable", False))
    raw_lines = parsed.get("lines", [])

    spans: list[ExtractedSpanSchema] = []
    flags: list[ExtractedFlagSchema] = []

    if isinstance(raw_lines, list):
        for idx, line_obj in enumerate(raw_lines):
            if not isinstance(line_obj, dict):
                continue
            text = str(line_obj.get("text", "")).strip()
            if not text:
                continue
            line_no = int(line_obj.get("lineNumber") or (len(spans) + 1))
            conf_raw = line_obj.get("confidence", 0.92)
            try:
                conf = max(0.0, min(1.0, round(float(conf_raw), 2)))
            except Exception:
                conf = 0.9
            span_id = f"{question_id}-L{line_no}-S1"
            spans.append(
                ExtractedSpanSchema(
                    id=span_id,
                    lineNumber=line_no,
                    text=text,
                    confidence=conf,
                )
            )
            unclear_note = str(line_obj.get("unclearNote") or "").strip()
            if "[?]" in text or conf < 0.78 or unclear_note:
                flags.append(
                    ExtractedFlagSchema(
                        type="UNCLEAR_CHAR" if "[?]" in text or unclear_note else "LOW_CONFIDENCE",
                        message=unclear_note or f"Line {line_no} extracted with reduced clarity — verify handwritten wording.",
                        spanId=span_id,
                    )
                )

    if not spans:
        blank_detected = True

    if blank_detected:
        flags.append(
            ExtractedFlagSchema(
                type="BLANK_ANSWER",
                message="No handwritten answer detected inside cropped region. Confirm Not Attempted or adjust crop.",
            )
        )

    if has_diagram:
        flags.append(
            ExtractedFlagSchema(
                type="UNSUPPORTED_CONTENT",
                message="Diagram, table, or non-linear sketch detected — verify or route for manual diagram review.",
            )
        )

    raw_text = "\n".join(sp.text for sp in spans)
    return active_model, StructuredQuestionResultSchema(
        questionId=question_id,
        rawText=raw_text,
        blankDetected=blank_detected,
        spans=spans,
        flags=flags,
    )


async def extract_structured_from_regions(
    regions: list[StructuredRegionPayload],
) -> tuple[str, list[StructuredQuestionResultSchema]]:
    """
    Runs pinned Gemini Vision OCR over cropped handwritten question regions,
    grouped by questionId in page order.
    """
    api_key = _resolve_gemini_api_key()
    model = _resolve_gemini_model()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in backend environment.")

    grouped: dict[str, list[StructuredRegionPayload]] = defaultdict(list)
    order: list[str] = []
    for reg in regions:
        if reg.questionId not in grouped:
            order.append(reg.questionId)
        grouped[reg.questionId].append(reg)

    used_model = model
    results: list[StructuredQuestionResultSchema] = []
    async with httpx.AsyncClient(timeout=40.0) as client:
        for q_id in order:
            used_model, res = await _extract_single_question_ocr(client, api_key, model, q_id, grouped[q_id])
            results.append(res)

    return used_model, results
