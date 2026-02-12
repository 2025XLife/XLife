"""
Safeguard Agent Models
Pydantic models for safety assessment input/output.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AssessmentStatus(str, Enum):
    """Assessment result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    REVIEW = "review"


class RiskFactor(BaseModel):
    """Individual risk factor identified"""
    factor: str = Field(..., description="Name of the risk factor")
    category: str = Field(
        ...,
        description="Category: medical, environmental, nutritional, exercise"
    )
    severity: RiskLevel = Field(..., description="Risk severity")
    description: str = Field(..., description="Detailed description")
    recommendation: str = Field(..., description="How to mitigate")


class SafetyCheck(BaseModel):
    """Individual safety check result"""
    check_name: str = Field(..., description="Name of the check")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(..., description="Result message")
    severity: Optional[RiskLevel] = Field(
        None,
        description="Severity if failed"
    )


class SafetyAssessment(BaseModel):
    """Complete safety assessment result"""
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Safety score (0-100)"
    )
    is_safe: bool = Field(
        ...,
        description="Whether the plan is safe to execute"
    )
    status: AssessmentStatus = Field(
        ...,
        description="Overall assessment status"
    )
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Identified risk factors"
    )
    safety_checks: List[SafetyCheck] = Field(
        default_factory=list,
        description="Detailed safety checks"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations to improve safety"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings for the user"
    )
    assessed_at: datetime = Field(
        default_factory=datetime.now,
        description="When the assessment was performed"
    )


class SafeguardInput(BaseModel):
    """Input structure for safeguard agent"""
    plan: Dict[str, Any] = Field(
        ...,
        description="The plan to assess (diet or exercise)"
    )
    plan_type: str = Field(
        ...,
        description="Type of plan: diet or exercise"
    )
    user_metadata: Dict[str, Any] = Field(
        ...,
        description="User physiological data"
    )
    environment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environmental context"
    )


class SafeguardResponse(BaseModel):
    """Response from safeguard agent"""
    assessment: SafetyAssessment = Field(
        ...,
        description="Safety assessment result"
    )
    plan_summary: Dict[str, Any] = Field(
        ...,
        description="Summary of the assessed plan"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next steps"
    )


class HealthPlanCombinedInput(BaseModel):
    """Combined input for diet + exercise + safeguard evaluation"""
    user_metadata: Dict[str, Any] = Field(
        ...,
        description="User physiological data"
    )
    environment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environmental context"
    )
    user_requirement: Dict[str, Any] = Field(
        default_factory=dict,
        description="User goals and preferences"
    )
    diet_plan: Optional[Dict[str, Any]] = Field(
        None,
        description="Diet plan to evaluate"
    )
    exercise_plan: Optional[Dict[str, Any]] = Field(
        None,
        description="Exercise plan to evaluate"
    )


class CombinedAssessment(BaseModel):
    """Combined assessment of diet and exercise plans"""
    diet_assessment: Optional[SafetyAssessment] = Field(
        None,
        description="Diet plan safety assessment"
    )
    exercise_assessment: Optional[SafetyAssessment] = Field(
        None,
        description="Exercise plan safety assessment"
    )
    overall_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Combined safety score"
    )
    is_combined_safe: bool = Field(
        ...,
        description="Whether combined plan is safe"
    )
    overall_recommendations: List[str] = Field(
        default_factory=list,
        description="Overall recommendations"
    )
    compatibility_notes: List[str] = Field(
        default_factory=list,
        description="Diet-exercise compatibility notes"
    )
