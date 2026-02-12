"""
Exercise Plan Parser
Parses and expands LLM-generated base exercise plans into multiple intensity variants.

Strategy: Generate "Lite", "Standard", "Plus" variants by applying
scaling rules based on duration and intensity.
"""
from typing import List, Dict, Any
from .models import ExerciseItem, ExerciseSession, ExercisePlan


class ExercisePlanParser:
    """
    Exercise plan parser and expander.

    Takes a base plan from LLM and expands it into multiple variants
    (Lite, Standard, Plus) based on deterministic scaling rules.
    """

    def __init__(self):
        # Scaling factors for each variant
        self.variants = {
            "Lite": 0.7,       # 70% of base duration
            "Standard": 1.0,    # 100% of base duration
            "Plus": 1.3        # 130% of base duration
        }

        # Intensity mappings for each variant
        self.intensity_map = {
            "Lite": {
                "very_high": "high",
                "high": "moderate",
                "moderate": "low",
                "low": "low"
            },
            "Standard": {
                "very_high": "very_high",
                "high": "high",
                "moderate": "moderate",
                "low": "low"
            },
            "Plus": {
                "very_high": "very_high",
                "high": "high",
                "moderate": "high",
                "low": "moderate"
            }
        }

    def expand_plan(
        self,
        base_plan: ExercisePlan,
        variants: List[str] = None
    ) -> Dict[str, ExercisePlan]:
        """
        Expand a base plan into multiple intensity variants.

        Args:
            base_plan: ExercisePlan from LLM
            variants: Which variants to generate (default: Lite, Standard, Plus)

        Returns:
            Dict mapping variant name to expanded ExercisePlan
            {
                "Lite": ExercisePlan(...with scaled durations),
                "Standard": ExercisePlan(...original),
                "Plus": ExercisePlan(...with increased durations)
            }
        """
        if variants is None:
            variants = ["Lite", "Standard", "Plus"]

        result = {}

        for variant_name in variants:
            scale_factor = self.variants.get(variant_name, 1.0)
            expanded_plan = self._scale_plan(base_plan, scale_factor, variant_name)
            result[variant_name] = expanded_plan

        return result

    def _scale_plan(
        self,
        base_plan: ExercisePlan,
        scale_factor: float,
        variant_name: str
    ) -> ExercisePlan:
        """
        Scale a complete exercise plan according to rules.

        Args:
            base_plan: ExercisePlan to scale
            scale_factor: Multiplier (e.g., 0.7, 1.0, 1.3)
            variant_name: Which variant (for intensity mapping)

        Returns:
            ExercisePlan with scaled durations and adjusted intensities
        """
        scaled_sessions = {}

        for time_key, session in base_plan.sessions.items():
            scaled_session = self._scale_session(session, scale_factor, variant_name)
            scaled_sessions[time_key] = scaled_session

        # Calculate totals
        total_duration = sum(
            s.total_duration_minutes for s in scaled_sessions.values()
        )
        total_calories = sum(
            s.total_calories_burned for s in scaled_sessions.values()
        )

        return ExercisePlan(
            id=base_plan.id,
            title=f"{base_plan.title} ({variant_name})",
            meal_timing=base_plan.meal_timing,
            sessions=scaled_sessions,
            total_duration_minutes=total_duration,
            total_calories_burned=total_calories,
            progression=base_plan.progression,
            reasoning=base_plan.reasoning,
            safety_notes=base_plan.safety_notes
        )

    def _scale_session(
        self,
        session: ExerciseSession,
        scale_factor: float,
        variant_name: str
    ) -> ExerciseSession:
        """
        Scale a single exercise session.

        Args:
            session: ExerciseSession to scale
            scale_factor: Duration multiplier
            variant_name: For intensity mapping

        Returns:
            ExerciseSession with scaled exercises
        """
        scaled_exercises = []
        intensity_map = self.intensity_map.get(variant_name, {})

        for exercise in session.exercises:
            # Scale duration
            new_duration = round(exercise.duration_minutes * scale_factor)
            new_duration = max(5, new_duration)  # Minimum 5 minutes per exercise

            # Scale calories proportionally
            if exercise.duration_minutes > 0:
                new_calories = round(
                    exercise.calories_burned * (new_duration / exercise.duration_minutes)
                )
            else:
                new_calories = exercise.calories_burned

            # Adjust intensity
            new_intensity = exercise.intensity
            if exercise.intensity in intensity_map:
                new_intensity = intensity_map[exercise.intensity]

            # Create scaled exercise
            scaled_exercise = ExerciseItem(
                name=exercise.name,
                exercise_type=exercise.exercise_type,
                duration_minutes=new_duration,
                intensity=new_intensity,
                calories_burned=new_calories,
                equipment=exercise.equipment,
                target_muscles=exercise.target_muscles,
                instructions=exercise.instructions,
                reason=exercise.reason,
                safety_notes=exercise.safety_notes
            )
            scaled_exercises.append(scaled_exercise)

        # Calculate session totals
        total_duration = sum(ex.duration_minutes for ex in scaled_exercises)
        total_calories = sum(ex.calories_burned for ex in scaled_exercises)

        # Determine overall intensity
        if scaled_exercises:
            intensities = [ex.intensity.value for ex in scaled_exercises]
            if "very_high" in intensities:
                overall = "very_high"
            elif "high" in intensities:
                overall = "high"
            elif "moderate" in intensities:
                overall = "moderate"
            else:
                overall = "low"
        else:
            overall = "low"

        return ExerciseSession(
            time_of_day=session.time_of_day,
            exercises=scaled_exercises,
            total_duration_minutes=total_duration,
            total_calories_burned=total_calories,
            overall_intensity=overall
        )

    def expand_single_plan(
        self,
        base_plan: ExercisePlan
    ) -> Dict[str, ExercisePlan]:
        """
        Convenience: Expand a single plan into all variants.

        Returns:
            Dict mapping variant name to scaled ExercisePlan
        """
        return self.expand_plan(base_plan)


# Convenience function
def expand_exercise_plan(
    base_plan: ExercisePlan
) -> Dict[str, ExercisePlan]:
    """
    Quick helper to expand an exercise plan.

    Args:
        base_plan: ExercisePlan from LLM

    Returns:
        Dict with "Lite", "Standard", "Plus" variants
    """
    parser = ExercisePlanParser()
    return parser.expand_plan(base_plan)
