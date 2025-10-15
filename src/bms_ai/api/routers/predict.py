from pydantic import BaseModel, Field
from sklearn import pipeline
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import pandas as pd

from src.bms_ai.api.dependencies import get_prescriptive_pipeline
from src.bms_ai.pipelines.prescriptive_pipeline import PrescriptivePipeline
from src.bms_ai.logger_config import setup_logger


log = setup_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="All required model input features including setpoints and other signals")


class PredictResponse(BaseModel):
    prediction: List[List[float]]
    input_used: Dict[str, Any]


@router.post("/", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    prescriptive_pipeline: PrescriptivePipeline = Depends(get_prescriptive_pipeline),
) -> PredictResponse:
    """Run a prediction using the loaded model and preprocessor.

    - Accepts a JSON object with all required input features (setpoints + others).
    - Uses the pipeline's preprocessor to transform and then model to predict.
    - Returns the raw prediction and echoes the input used.
    """
    print(f"in predict endpoint, req.features keys: {list(req.features.keys())}")
    try:
        if not prescriptive_pipeline.model or not prescriptive_pipeline.preprocessor:
            raise HTTPException(status_code=500, detail="Model or preprocessor not loaded.")
        print("Here")
        input_df = pd.DataFrame([req.features])
        try:
            input_df_reordered = input_df[prescriptive_pipeline.preprocessor.feature_names_in_]
            print("Input DataFrame reordered successfully.")
        except Exception as e:
            missing = [c for c in getattr(prescriptive_pipeline.preprocessor, 'feature_names_in_', []) if c not in input_df.columns]
            log.error(f"Input missing required features: {missing}. Error: {e}")
            raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

        try:
            transformed = prescriptive_pipeline.preprocessor.transform(input_df_reordered)
            feature_out = prescriptive_pipeline.preprocessor.get_feature_names_out()
            transformed_df = pd.DataFrame(transformed, columns=feature_out)
            print("Preprocessing successful.")
        except Exception as e:
            log.error(f"Preprocessing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

        try:
            prediction = prescriptive_pipeline.model.predict(transformed_df)
            print("Prediction successful.")
        except Exception as e:
            log.error(f"Model prediction failed: {e}")
            raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

        # Convert to JSON-serializable list
        return PredictResponse(prediction=prediction.tolist(), input_used=req.features)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Prediction request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
