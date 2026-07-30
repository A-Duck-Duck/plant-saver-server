"""
Plant Saver API

Endpoint:
    POST /diagnose
        form-data:
            plant_image: file (required)  -- photo of the whole plant
            leaf_image:  file (optional)  -- closeup of a specific leaf
            deep_dive:   bool (optional, default false) -- whether to call Claude
                         for an explanation + care steps. Off by default until
                         an ANTHROPIC_API_KEY is set up.

Run locally with:
    uvicorn app:app --reload
"""

import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from model_utils import predict_combined
from claude_helper import dive_deeper

app = FastAPI(title="Plant Saver API")


@app.get("/")
def root():
    return {"status": "Plant Saver API is running"}


@app.post("/diagnose")
async def diagnose(
    plant_image: UploadFile = File(...),
    leaf_image: UploadFile = File(None),
    deep_dive: bool = Form(False),
):
    try:
        plant_bytes = io.BytesIO(await plant_image.read())
        leaf_bytes = io.BytesIO(await leaf_image.read()) if leaf_image else None

        result = predict_combined(plant_bytes, leaf_bytes)

        response = {
            "diagnosis": result["label"],
            "confidence": round(result["confidence"], 4),
            "all_scores": {k: round(v, 4) for k, v in result["all_scores"].items()},
            "used_leaf_photo": leaf_bytes is not None,
        }

        if deep_dive:
            response["deep_dive"] = dive_deeper(
                result["label"], result["confidence"], result["all_scores"]
            )

        return JSONResponse(content=response)

    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Something went wrong: {e}"})
