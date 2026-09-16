from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..schemas import PredictRequest
from ..services.ml_service import predict_sample

router = APIRouter(prefix="/api/runs", tags=["Inference & Prediction"])


@router.post("/{run_id}/predict")
def predict(run_id: int, payload: PredictRequest, db: Session = Depends(get_db)):
    """Run model inference on single sample feature dictionary."""
    try:
        res = predict_sample(run_id, payload.features, db)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
