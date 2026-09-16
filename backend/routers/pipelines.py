from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..schemas import PreviewRecipeRequest
from ..services.ml_service import preview_recipe

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
