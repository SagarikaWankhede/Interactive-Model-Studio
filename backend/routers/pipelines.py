from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..database import get_db
from ..schemas import PreviewRecipeRequest
from ..services.ml_service import preview_recipe, get_available_algorithms
from ml_engine.pipeline_executor import clear_pipeline_cache

router = APIRouter(prefix="/api/pipelines", tags=["Pipelines"])


@router.post("/preview")
def preview_pipeline_recipe(payload: PreviewRecipeRequest, db: Session = Depends(get_db)):
    """Preview preprocessing transformations on a dataset sample without fitting models."""
    try:
        recipe_dicts = [step.model_dump() for step in payload.recipe]
        result = preview_recipe(
            dataset_id=payload.dataset_id,
            target_column=payload.target_column,
            recipe=recipe_dicts,
            db=db
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing preview failed: {str(e)}")


@router.get("/algorithms")
@router.get("/models")
def get_supported_algorithms(task_type: Optional[str] = Query(None, description="classification or regression")):
    """Returns available algorithms, metadata, and default hyperparameters from ml_engine.registry."""
    try:
        return {"algorithms": get_available_algorithms(task_type)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch algorithms: {str(e)}")


@router.post("/cache/clear")
def clear_cache():
    """Clears the preprocessing transform cache."""
    clear_pipeline_cache()
    return {"status": "success", "message": "Pipeline transform cache cleared successfully."}
