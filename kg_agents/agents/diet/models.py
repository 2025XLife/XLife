"""
Diet Agent Models
Pydantic models for diet recommendation input/output.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


# Enums & Constants

class MealType(str, Enum):
    """Types of meals in a day"""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACKS = "snacks"


ALLOWED_UNITS = Literal["gram", "ml", "piece", "slice", "cup", "bowl", "spoon"]


class BaseFoodItem(BaseModel):
    """LLM output: Base food item with standardized units for parser expansion"""
    food_name: str = Field(..., description="Name of the food dish")
    portion_number: float = Field(..., description="Numeric quantity (e.g., 100, 1.5)")
    portion_unit: ALLOWED_UNITS = Field(..., description="Unit: gram, ml, piece, slice, cup, bowl, or spoon")
    total_calories: Optional[float] = Field(None, description="Total calories for this portion size (preferred)")
    calories_per_unit: Optional[float] = Field(None, description="Legacy: calories per unit (deprecated, use total_calories)")

    model_config = {
        "populate_by_name": True
    }


class RawDietPlan(BaseModel):
    """LLM output: Raw diet plan containing base food items"""
    items: List[BaseFoodItem] = Field(..., description="List of food items in this meal")


# ================= Existing Models (for downstream processing) =================


class FoodItem(BaseModel):
    """A single food item in a meal"""
    food: str = Field(..., description="Name of the food dish")
    portion: str = Field(..., description="Portion size (e.g., '100g', '1bowl', '2piece')")
    calories: int = Field(..., description="Estimated calories per serving")
    # protein: float = Field(..., description="Protein content in grams")
    # carbs: float = Field(..., description="Carbohydrate content in grams")
    # fat: float = Field(..., description="Fat content in grams")
    # fiber: Optional[float] = Field(None, description="Fiber content in grams")


class MealPlanItem(BaseModel):
    """A complete meal with multiple food items"""
    meal_type: MealType = Field(..., description="Type of meal")
    items: List[FoodItem] = Field(..., description="Food items in this meal")
    total_calories: int = Field(..., description="Total calories for this meal")
    total_protein: float = Field(..., description="Total protein in grams")
    total_carbs: float = Field(..., description="Total carbohydrates in grams")
    total_fat: float = Field(..., description="Total fat in grams")


class MacroNutrients(BaseModel):
    """Daily macro nutrient summary"""
    protein: float = Field(..., description="Total protein in grams")
    carbs: float = Field(..., description="Total carbohydrates in grams")
    fat: float = Field(..., description="Total fat in grams")
    protein_ratio: float = Field(..., description="Protein calorie percentage (15-25% ideal)")
    carbs_ratio: float = Field(..., description="Carbohydrate calorie percentage (45-65% ideal)")
    fat_ratio: float = Field(..., description="Fat calorie percentage (20-35% ideal)")


class DietRecommendation(BaseModel):
    """Diet recommendation for a single meal with multiple portion variants"""
    id: int = Field(..., description="Candidate ID")
    meal_type: MealType = Field(..., description="Which meal this is for (breakfast/lunch/dinner/snacks)")
    variant: str = Field(..., description="Portion variant: Lite/Standard/Plus")
    items: List[FoodItem] = Field(..., description="Food items in this meal")
    total_calories: int = Field(..., description="Total calories for this meal")
    target_calories: int = Field(..., description="Target calories for this meal")
    calories_deviation: float = Field(
        ...,
        description="Deviation from target calories (%)"
    )
    safety_notes: List[str] = Field(
        default_factory=list,
        description="Safety considerations"
    )


class DietCandidatesResponse(BaseModel):
    """Response containing multiple diet candidates"""
    candidates: List[DietRecommendation] = Field(
        ...,
        description="List of diet candidates"
    )
    target_calories: int = Field(..., description="Target daily calories")
    user_conditions: List[str] = Field(
        ...,
        description="User's health conditions considered"
    )
    sampling_strategy: str = Field(
        ...,
        description="Strategy used for generation"
    )
    generation_notes: str = Field(
        ...,
        description="Additional notes about generation"
    )


class DietAgentInput(BaseModel):
    """Input structure for diet agent"""
    user_metadata: Dict[str, Any] = Field(
        ...,
        description="User physiological data"
    )
    environment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environmental context (weather, season)"
    )
    user_requirement: Dict[str, Any] = Field(
        default_factory=dict,
        description="User goals and preferences"
    )
    num_candidates: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of candidates to generate"
    )
