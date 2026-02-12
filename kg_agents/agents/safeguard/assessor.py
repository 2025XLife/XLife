import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from agents.base import BaseAgent
from agents.safeguard.models import (
    RiskLevel, AssessmentStatus,
    RiskFactor, SafetyCheck, SafetyAssessment,
    SafeguardInput, SafeguardResponse
)
from core.llm import get_llm
from core.llm.utils import parse_json_response
from core.neo4j import get_kg_query
from agents.safeguard.config import *
from core.prompts import (
    DIETARY_QUERY_ENTITIES,
    get_keywords
)


DIET_SAFETY_RULES = get_DIET_SAFETY_RULES(RiskLevel)
EXERCISE_SAFETY_RULES = get_EXERCISE_SAFETY_RULES(RiskLevel)
CONDITION_RESTRICTIONS = get_CONDITION_RESTRICTIONS()

SAFETY_MEASURE = 2
# SAFETY_MEASURE = 1: score based (current implementation)
# SAFETY_MEASURE = 2: LLM risk_factors.severity based 
# SAFETY_MEASURE = 3: LLM checks.passed based 


# security agent

class SafeguardAgent(BaseAgent):
    def get_agent_name(self) -> str:
        return "safeguard"

    def get_input_type(self):
        return SafeguardInput

    def get_output_type(self):
        return SafetyAssessment

    def generate(
        self,
        input_data: Dict[str, Any],
        num_candidates: int = 1
    ) -> List[SafetyAssessment]:
        plan = input_data.get("plan", {})
        plan_type = input_data.get("plan_type", "diet")
        user_metadata = input_data.get("user_metadata", {})
        environment = input_data.get("environment", {})

        assessment = self.assess(plan, plan_type, user_metadata, environment)
        return [assessment]

    def assess(
        self,
        plan: Dict[str, Any],
        plan_type: str,
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any] = {}
    ) -> SafetyAssessment:
        checks = []
        risk_factors = []

        # --- MODIFICATION START: Conditional Rule Execution ---
        if ENABLE_RULE_BASED_CHECKS:
            # 1. Run rule-based checks
            if plan_type == "diet":
                rule_checks, rule_risks = self._check_diet_safety(
                    plan, user_metadata
                )
            elif plan_type == "exercise":
                rule_checks, rule_risks = self._check_exercise_safety(
                    plan, user_metadata
                )
            else:
                rule_checks, rule_risks = [], []
            checks.extend(rule_checks)
            risk_factors.extend(rule_risks)

            # 2. Run condition-specific checks
            condition_checks, condition_risks = self._check_condition_restrictions(
                plan, plan_type, user_metadata
            )
            checks.extend(condition_checks)
            risk_factors.extend(condition_risks)

            # 3. Run environment checks
            env_checks, env_risks = self._check_environment_safety(
                plan, plan_type, environment
            )
            checks.extend(env_checks)
            risk_factors.extend(env_risks)
        # --- MODIFICATION END ---

        # LLM semantic assessment (Always run or run if rules disabled)
        # print(f"LLM Assessing...")
        llm_assessment = self._llm_semantic_assessment(
            plan, plan_type, user_metadata, environment
        )

        # Merge LLM findings BEFORE scoring
        if llm_assessment:
            for rf_dict in llm_assessment.get("risk_factors", []):
                if isinstance(rf_dict, dict):
                    risk_factors.append(RiskFactor(**rf_dict))
            for check_dict in llm_assessment.get("checks", []):
                if isinstance(check_dict, dict):
                    checks.append(SafetyCheck(**check_dict))

        # --- MOVED SCORING LOGIC HERE ---

        if ENABLE_RULE_BASED_CHECKS and SAFETY_MEASURE == 1:
            # Calculate score based on ALL checks (Rules + LLM)
            passed_checks = sum(1 for c in checks if c.passed)
            total_checks = len(checks) if checks else 1  # Avoid div by zero

            # Base score
            if not checks:
                # If no checks ran at all (e.g. LLM failed and rules disabled), assume neutral/safe
                base_score = 100
            else:
                base_score = (passed_checks / total_checks) * 100

            # Apply severity penalties from ALL risk factors (Rules + LLM)
            severity_penalty = 0
            for rf in risk_factors:
                penalty = {"low": 5, "moderate": 15, "high": 30, "very_high": 50}
                severity_penalty += penalty.get(rf.severity.value, 0)

            # Final score calculation
            final_score = max(0, min(100, base_score - severity_penalty))

            # Determine if safe
            is_safe = final_score >= 60 and not any(
                rf.severity in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
                for rf in risk_factors
            )

            # Determine status
            if final_score >= 80:
                status = AssessmentStatus.PASSED
            elif final_score >= 60:
                status = AssessmentStatus.WARNING
            elif final_score >= 40:
                status = AssessmentStatus.REVIEW
            else:
                status = AssessmentStatus.FAILED

            # Determine risk level
            if final_score >= 80:
                risk_level = RiskLevel.LOW
            elif final_score >= 60:
                risk_level = RiskLevel.MODERATE
            elif final_score >= 40:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.VERY_HIGH

            # Generate recommendations
            recommendations = self._generate_recommendations(
                risk_factors, plan_type, user_metadata
            )

            return SafetyAssessment(
                score=final_score,
                is_safe=is_safe,
                status=status,
                risk_level=RiskLevel.LOW,
                risk_factors=risk_factors,
                safety_checks=checks,
                recommendations=recommendations,
                warnings=[rf.description for rf in risk_factors if rf.severity in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]]
            )
        elif SAFETY_MEASURE == 2:
            # print(f"Evaluate LLM Assessment by risk factors...")
            passed = True
            for rf in risk_factors:
                penalty_values = ["high", "very high"]
                # penalty_values = ["very high"]
                if rf.severity.value in penalty_values:
                    passed = False
                    break
            if passed:
                return SafetyAssessment(
                    score=100,
                    is_safe=True,
                    status=AssessmentStatus.PASSED,
                    risk_level=RiskLevel.LOW,
                    risk_factors=risk_factors,
                    safety_checks=checks,
                    recommendations=[],
                    warnings=[rf.description for rf in risk_factors if rf.severity in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]]
                )
            else:
                return SafetyAssessment(
                    score=0,
                    is_safe=False,
                    status=AssessmentStatus.FAILED,
                    risk_level=RiskLevel.VERY_HIGH,
                    risk_factors=risk_factors,
                    safety_checks=checks,
                    recommendations=[],
                    warnings=[rf.description for rf in risk_factors if rf.severity in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]]
                )

        elif SAFETY_MEASURE == 3:
            print(f"Evaluate LLM Assessment by checks...")
            passed = True
            for c in checks:
                if not c.passed:
                    passed = False
                    break
            if passed:
                return SafetyAssessment(
                    score=100,
                    is_safe=True,
                    status=AssessmentStatus.PASSED,
                    risk_level=RiskLevel.LOW,
                    risk_factors=risk_factors,
                    safety_checks=checks,
                    recommendations=[],
                    warnings=[rf.description for rf in risk_factors if rf.severity in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]]
                )
            else:
                return SafetyAssessment(
                    score=0,
                    is_safe=False,
                    status=AssessmentStatus.FAILED,
                    risk_level=RiskLevel.VERY_HIGH,
                    risk_factors=risk_factors,
                    safety_checks=checks,
                    recommendations=[],
                    warnings=[rf.description for rf in risk_factors if rf.severity in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]]
                )
        else:
            raise ValueError("Invalid ENABLE_RULE_BASED_CHECKS")

    def _check_diet_safety(
        self,
        plan: Dict[str, Any],
        user_metadata: Dict[str, Any]
    ) -> tuple:
        checks = []
        risk_factors = []

        # DietRecommendation.total_calories (int)
        total_calories = plan.get("total_calories", 0)

        # Calorie range check
        if total_calories < DIET_SAFETY_RULES["min_calories"]["value"]:
            checks.append(SafetyCheck(
                check_name="min_calories",
                passed=False,
                message=DIET_SAFETY_RULES["min_calories"]["message"],
                severity=RiskLevel.HIGH
            ))
            risk_factors.append(RiskFactor(
                factor="extremely_low_calories",
                category="nutritional",
                severity=RiskLevel.HIGH,
                description=f"Total calories {total_calories} is dangerously low",
                recommendation="Consult a dietitian for safe calorie targets"
            ))
        elif total_calories > DIET_SAFETY_RULES["max_calories"]["value"]:
            checks.append(SafetyCheck(
                check_name="max_calories",
                passed=False,
                message=DIET_SAFETY_RULES["max_calories"]["message"],
                severity=RiskLevel.MODERATE
            ))
        else:
            checks.append(SafetyCheck(
                check_name="calories_range",
                passed=True,
                message="Calorie intake within acceptable range"
            ))

        macros = plan.get("macro_nutrients", {})
        if macros:
            if isinstance(macros, dict):
                protein_ratio = macros.get("protein_ratio", 0)
                fat_ratio = macros.get("fat_ratio", 0)
            else:
                protein_ratio = getattr(macros, "protein_ratio", 0)
                fat_ratio = getattr(macros, "fat_ratio", 0)

            if protein_ratio < DIET_SAFETY_RULES["min_protein_ratio"]["value"]:
                risk_factors.append(RiskFactor(
                    factor="low_protein",
                    category="nutritional",
                    severity=RiskLevel.MODERATE,
                    description=f"Protein ratio {protein_ratio*100:.1f}% is below recommended minimum",
                    recommendation="Include more protein-rich foods"
                ))

            if fat_ratio > DIET_SAFETY_RULES["max_fat_ratio"]["value"]:
                risk_factors.append(RiskFactor(
                    factor="high_fat",
                    category="nutritional",
                    severity=RiskLevel.MODERATE,
                    description=f"Fat ratio {fat_ratio*100:.1f}% exceeds recommended maximum",
                    recommendation="Reduce high-fat foods"
                ))

        meal_plan = plan.get("meal_plan", {})
        if meal_plan:
            for meal_type, items in meal_plan.items():
                if isinstance(items, list):
                    meal_calories = sum(item.get("calories", 0) for item in items if isinstance(item, dict))
                elif isinstance(items, dict):
                    # Might be a MealPlanItem object
                    meal_calories = items.get("total_calories", 0)
                else:
                    meal_calories = 0

                if meal_calories > DIET_SAFETY_RULES["single_meal_calories"]["value"]:
                    checks.append(SafetyCheck(
                        check_name="single_meal_calories",
                        passed=False,
                        message=f"{meal_type} calorie {meal_calories} is high",
                        severity=RiskLevel.LOW
                    ))

        return checks, risk_factors

    def _check_exercise_safety(
        self,
        plan: Dict[str, Any],
        user_metadata: Dict[str, Any]
    ) -> tuple:
        checks = []
        risk_factors = []

        # ExercisePlan fields
        fitness_level = user_metadata.get("fitness_level", "beginner")
        total_duration = plan.get("total_duration_minutes", 0)
        sessions = plan.get("sessions", {})
        weekly_freq = plan.get("weekly_frequency", 3)

        # Duration limits based on fitness level
        max_duration_map = {
            "beginner": EXERCISE_SAFETY_RULES["max_daily_duration_beginner"]["value"],
            "intermediate": EXERCISE_SAFETY_RULES["max_daily_duration_intermediate"]["value"],
            "advanced": EXERCISE_SAFETY_RULES["max_daily_duration_advanced"]["value"]
        }
        max_duration = max_duration_map.get(fitness_level, 60)

        if total_duration > max_duration:
            severity = RiskLevel.MODERATE if fitness_level == "advanced" else RiskLevel.HIGH
            checks.append(SafetyCheck(
                check_name="daily_duration",
                passed=False,
                message=f"Duration {total_duration}min exceeds {fitness_level} limit ({max_duration}min)",
                severity=severity
            ))
            risk_factors.append(RiskFactor(
                factor="excessive_duration",
                category="exercise",
                severity=severity,
                description=f"Total exercise time {total_duration}min is excessive for {fitness_level}",
                recommendation=f"Reduce daily duration to {max_duration}min or less"
            ))
        else:
            checks.append(SafetyCheck(
                check_name="daily_duration",
                passed=True,
                message=f"Duration {total_duration}min is appropriate"
            ))

        # Rest day check
        if weekly_freq > EXERCISE_SAFETY_RULES["max_weekly_sessions"]["value"]:
            checks.append(SafetyCheck(
                check_name="rest_days",
                passed=False,
                message="Exercise every day without rest",
                severity=RiskLevel.MODERATE
            ))
            risk_factors.append(RiskFactor(
                factor="no_rest_days",
                category="exercise",
                severity=RiskLevel.MODERATE,
                description="No rest days scheduled in weekly plan",
                recommendation="Include at least 1-2 rest days per week"
            ))

        # HIIT frequency check - handle both string and enum formats
        def get_exercise_type(ex):
            """Extract exercise type, handling both string and enum formats"""
            et = ex.get("exercise_type", "")
            if isinstance(et, str):
                return et.lower()
            # Handle enum
            return str(et).lower() if hasattr(et, 'value') else str(et).lower()

        has_hiit = False
        for session_key, session in sessions.items():
            # session can be dict or ExerciseSession object
            exercises = session.get("exercises", []) if isinstance(session, dict) else []
            if not exercises and hasattr(session, 'exercises'):
                exercises = session.exercises

            for ex in exercises:
                if isinstance(ex, dict):
                    ex_type = get_exercise_type(ex)
                else:
                    # Object format
                    ex_type = str(ex.exercise_type).lower() if hasattr(ex, 'exercise_type') else ""
                if ex_type == "hiit":
                    has_hiit = True
                    break
            if has_hiit:
                break

        if has_hiit and weekly_freq > 3:
            risk_factors.append(RiskFactor(
                factor="hiit_frequency",
                category="exercise",
                severity=RiskLevel.HIGH,
                description="HIIT sessions too frequent without adequate recovery",
                recommendation="Limit HIIT to 2-3 times per week with 48h rest"
            ))

        # High intensity check for beginners/intermediate
        if fitness_level in ["beginner", "intermediate"]:
            has_high_intensity = False
            for session_key, session in sessions.items():
                exercises = session.get("exercises", []) if isinstance(session, dict) else []
                if not exercises and hasattr(session, 'exercises'):
                    exercises = session.exercises

                for ex in exercises:
                    if isinstance(ex, dict):
                        intensity = str(ex.get("intensity", "")).lower()
                    else:
                        intensity = str(ex.intensity).lower() if hasattr(ex, 'intensity') else ""
                    if intensity == "high" or intensity == "very_high":
                        has_high_intensity = True
                        break
                if has_high_intensity:
                    break

            if has_high_intensity:
                checks.append(SafetyCheck(
                    check_name="intensity_level",
                    passed=False,
                    message=f"High intensity exercise may not be suitable for {fitness_level}",
                    severity=RiskLevel.MODERATE
                ))

        return checks, risk_factors

    def _extract_plan_content_text(self, plan: Dict[str, Any], plan_type: str) -> str:
        """Extract readable text from plan for condition checking"""
        content_parts = []

        if plan_type == "diet":
            # Extract food items
            meal_plan = plan.get("meal_plan", {})
            if isinstance(meal_plan, dict):
                for meal_type, items in meal_plan.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                food_name = item.get("food", "")
                                if food_name:
                                    content_parts.append(food_name)
                            elif hasattr(item, 'food'):
                                content_parts.append(item.food)
            # Add macro info
            macros = plan.get("macro_nutrients", {})
            if isinstance(macros, dict):
                content_parts.extend([str(v) for v in macros.values()])

        elif plan_type == "exercise":
            # Extract exercise names and types
            sessions = plan.get("sessions", {})
            if isinstance(sessions, dict):
                for session_key, session in sessions.items():
                    if isinstance(session, dict):
                        exercises = session.get("exercises", [])
                        for ex in exercises:
                            if isinstance(ex, dict):
                                ex_name = ex.get("name", "")
                                ex_type = ex.get("exercise_type", "")
                                if ex_name:
                                    content_parts.append(ex_name)
                                if ex_type:
                                    content_parts.append(str(ex_type))
                            elif hasattr(ex, 'name'):
                                content_parts.append(ex.name)
                                if hasattr(ex, 'exercise_type'):
                                    content_parts.append(str(ex.exercise_type))

        return " ".join(content_parts).lower()

    def _check_condition_restrictions(
        self,
        plan: Dict[str, Any],
        plan_type: str,
        user_metadata: Dict[str, Any]
    ) -> tuple:
        checks = []
        risk_factors = []

        conditions = user_metadata.get("medical_conditions", [])

        plan_content = self._extract_plan_content_text(plan, plan_type)

        for condition in conditions:
            condition_lower = condition.lower()
            matched_condition = None
            for known_condition in CONDITION_RESTRICTIONS:
                if condition_lower == known_condition or condition_lower.replace("_", "") == known_condition.replace("_", ""):
                    matched_condition = known_condition
                    break

            if matched_condition:
                restrictions = CONDITION_RESTRICTIONS[matched_condition]

                # Check diet restrictions
                if plan_type == "diet" and "diet" in restrictions:
                    diet_rules = restrictions["diet"]

                    for rule_key, rule_desc in diet_rules.items():
                        if "avoid" in rule_key or "max" in rule_key:
                            # Check if plan contains forbidden items
                            forbidden_keywords = {
                                "high_sugar": ["sugar", "candy", "cake", "ice cream", "sweet"],
                                "high_sodium": ["salt", "soy sauce", "pickled", "processed"],
                                "high_cholesterol": ["egg yolk", "liver", "organ meat"],
                                "caffeine": ["coffee", "tea", "caffeine"],
                            }
                            keywords = forbidden_keywords.get(rule_key, [])
                            for keyword in keywords:
                                if keyword in plan_content:
                                    risk_factors.append(RiskFactor(
                                        factor=f"{matched_condition}_{rule_key}",
                                        category="medical",
                                        severity=RiskLevel.HIGH,
                                        description=f"Plan contains {keyword} which should be avoided for {condition}",
                                        recommendation=f"Remove {rule_desc} for {condition} management"
                                    ))

                # Check exercise restrictions
                if plan_type == "exercise" and "exercise" in restrictions:
                    ex_rules = restrictions["exercise"]

                    for rule_key, rule_desc in ex_rules.items():
                        if "avoid" in rule_key or "max" in rule_key or "isometric" in rule_key:
                            forbidden_exercises = {
                                "isometric": ["plank", "wall sit", "static hold"],
                                "high_intensity": ["hiit", "sprint", "burpee", "jump"],
                                "high_impact": ["running", "jumping", "jump rope"],
                                "breath_holding": ["valsalva", "heavy lifting"],
                            }
                            keywords = forbidden_exercises.get(rule_key, [])
                            for keyword in keywords:
                                if keyword in plan_content:
                                    risk_factors.append(RiskFactor(
                                        factor=f"{matched_condition}_{rule_key}",
                                        category="medical",
                                        severity=RiskLevel.HIGH,
                                        description=f"Exercise plan contains {keyword} which violates {condition} restriction",
                                        recommendation=f"Modify plan to comply with {rule_desc}"
                                    ))

        return checks, risk_factors

    def _check_environment_safety(
        self,
        plan: Dict[str, Any],
        plan_type: str,
        environment: Dict[str, Any]
    ) -> tuple:
        """Check plan against environmental factors"""
        checks = []
        risk_factors = []

        weather = environment.get("weather", {})
        condition = weather.get("condition", "clear")
        temperature = weather.get("temperature_c", 20)

        # Temperature-based checks for outdoor exercise
        if plan_type == "exercise":
            if temperature > 35:
                risk_factors.append(RiskFactor(
                    factor="high_temperature_exercise",
                    category="environmental",
                    severity=RiskLevel.HIGH,
                    description=f"High temperature ({temperature} degrees Celsius) increases heat stress risk",
                    recommendation="Exercise indoors or in early morning/late evening"
                ))
            elif temperature < 5:
                risk_factors.append(RiskFactor(
                    factor="cold_temperature_exercise",
                    category="environmental",
                    severity=RiskLevel.MODERATE,
                    description=f"Cold temperature ({temperature} degrees Celsius) increases cardiovascular strain",
                    recommendation="Warm up thoroughly, dress in layers"
                ))

            if condition == "rainy" or condition == "icy":
                risk_factors.append(RiskFactor(
                    factor="inclement_weather",
                    category="environmental",
                    severity=RiskLevel.MODERATE,
                    description=f"{condition} weather increases slip/fall risk",
                    recommendation="Move exercise indoors or choose safe surfaces"
                ))

        # Diet considerations for environment
        if plan_type == "diet":
            if temperature > 30:
                # Suggest lighter meals
                checks.append(SafetyCheck(
                    check_name="hot_weather_hydration",
                    passed=True,
                    message="Consider increased fluid intake for hot weather"
                ))

        return checks, risk_factors

    def _llm_semantic_assessment(
        self,
        plan: Dict[str, Any],
        plan_type: str,
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for semantic safety assessment with KG context"""
        try:
            # Build KG context if available
            kg_context = ""
            if plan_type == "diet":
                kg_context = self._query_diet_kg_for_assessment(plan, user_metadata)
            elif plan_type == "exercise":
                kg_context = self._query_exercise_kg_for_assessment(plan, user_metadata)

#             prompt = f"""Analyze the following {plan_type} plan for safety issues.

# ## User Profile
# - Age: {user_metadata.get('age', 'unknown')}
# - Conditions: {', '.join(user_metadata.get('medical_conditions', ['none']))}
# - Fitness Level: {user_metadata.get('fitness_level', 'unknown')}

# ## Environment
# {environment}

# ## Knowledge Graph Guidelines
# {kg_context if kg_context else "No KG data available"}

# ## Plan
# {json.dumps(plan, ensure_ascii=False, indent=2)}

# ## Task
# Identify any safety concerns:
# 1. Hidden contraindications
# 2. Unrealistic progression
# 3. Nutrient deficiencies
# 4. Overtraining signs
# 5. Environmental mismatches
# 6. Conflicts with user's medical conditions

# ## Output Format (STRICT JSON)
# Return a single valid JSON object containing two lists: "risk_factors" and "checks".
# Follow the schema definitions below STRICTLY.

# 1. "risk_factors": list of objects containing:
#    - "factor": (string) Name of the risk factor
#    - "category": (string) Must be one of ["medical", "environmental", "nutritional", "exercise"]
#    - "severity": (string) Must be one of ["low", "moderate", "high", "very_high"]
#    - "description": (string) Detailed description of the risk
#    - "recommendation": (string) Actionable mitigation advice

# 2. "checks": list of objects containing:
#    - "check_name": (string) Name of the specific check performed
#    - "passed": (boolean) true or false
#    - "message": (string) Explanation of the check result
#    - "severity": (string, optional) If passed is false, must be one of ["low", "moderate", "high", "very_high"]

# Ensure "severity" values matches the allowed Enum values EXACTLY.
# """
            # Assess Prompt
            prompt = f"""Analyze the following {plan_type} plan for safety issues.

## Profile:
{user_metadata}

## Environment:
{environment}

## Knowledge Graph Guidelines:
{kg_context if kg_context else "No KG data available"}

## Plan:
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Task:
Identify any safety concerns:
1. Hidden contraindications
2. Unrealistic progression
3. Nutrient deficiencies
4. Overtraining signs
5. Environmental mismatches
6. Conflicts with user's medical conditions

## Output Format (STRICT JSON):
Return a single valid JSON object containing two lists: "risk_factors" and "checks".
Follow the schema definitions below STRICTLY.

1. "risk_factors": list of objects containing:
   - "factor": (string) Name of the risk factor
   - "category": (string) Must be one of ["medical", "environmental", "nutritional", "exercise"]
   - "severity": (string) Must be one of ["low", "moderate", "high", "very_high"]
   - "description": (string) Detailed description of the risk
   - "recommendation": (string) Actionable mitigation advice

2. "checks": list of objects containing:
   - "check_name": (string) Name of the specific check performed
   - "passed": (boolean) true or false
   - "message": (string) Explanation of the check result
   - "severity": (string, optional) If passed is false, must be one of ["low", "moderate", "high", "very_high"]

Ensure "severity" values matches the allowed Enum values EXACTLY.
"""

            response = self._call_llm(
                system_prompt="You are a safety assessment expert. ",
                user_prompt=prompt,
                temperature=0.3
            )

            if isinstance(response, str):
                return parse_json_response(response)
            return response

        except Exception as e:
            print(f"LLM assessment failed: {e}")
            return None

    def _query_diet_kg_for_assessment(
        self,
        plan: Dict[str, Any],
        user_metadata: Dict[str, Any],
        use_vector_search: bool = True  # GraphRAG: use vector search instead of keyword matching
    ) -> str:
        """
        Query knowledge graph for diet plan safety assessment using GraphRAG approach

        Args:
            plan: Diet plan with food items
            user_metadata: User metadata including conditions
            use_vector_search: If True, use vector search (GraphRAG); else use keyword matching
        """
        kg = get_kg_query()
        results = []

        # Extract food items from plan
        food_items = []
        meal_plan_str = ""
        meal_plan_names = ""
        meal_plan = plan.get("meal_plan", {})
        if isinstance(meal_plan, dict):
            items = meal_plan.get("items", [])
            for item in items:
                food = item.get("food", "")
                portion = item.get("portion", "")
                meal_plan_names += food
                meal_plan_str += f"{portion} of {food}; "

        # Get user conditions and restrictions
        conditions = user_metadata.get("medical_conditions", [])
        restrictions = user_metadata.get("dietary_restrictions", [])

        # === GraphRAG Approach: Vector Search + Graph Traversal ===
        use_vector_search = True
        if use_vector_search:
            try:
                # 1. For each food item, use vector search to find similar entities
                seen_entities = set()
                for food_item in food_items[:5]:  # Limit to 5 food items
                    anchors = kg.search_similar_entities(food_item, top_k=2)

                    for anchor in anchors:
                        anchor_name = anchor.get("name", "")
                        if not anchor_name:
                            continue

                        if anchor_name not in seen_entities:
                            seen_entities.add(anchor_name)
                            # Get neighbors for graph traversal (1-hop)
                            neighbors = kg.client.get_neighbors(anchor_name)

                            for neighbor in neighbors:
                                entity_name = neighbor.get("neighbor", "")
                                rel_type = neighbor.get("rel_type", "")
                                condition = neighbor.get("condition", "")

                                if not entity_name:
                                    continue

                                # Filter by prioritized risk relations
                                # if rel_type not in prioritized_risk_kg_rels:
                                #     continue

                                results.append({
                                    "entity": anchor_name,
                                    "relation": rel_type,
                                    "related_to": entity_name,
                                    "condition": condition
                                })

                # 2. Also add conditions and restrictions
                for condition in conditions + restrictions:
                    anchors = kg.search_similar_entities(condition, top_k=2)
                    for anchor in anchors:
                        anchor_name = anchor.get("name", "")
                        if not anchor_name:
                            continue

                        neighbors = kg.client.get_neighbors(anchor_name)
                        for neighbor in neighbors:
                            entity_name = neighbor.get("neighbor", "")
                            rel_type = neighbor.get("rel_type", "")
                            condition = neighbor.get("condition", "")

                            if not entity_name:
                                continue

                            # if rel_type not in prioritized_risk_kg_rels:
                            #     continue

                            results.append({
                                "entity": anchor_name,
                                "relation": rel_type,
                                "related_to": entity_name,
                                "condition": condition
                            })

            except Exception as e:
                print(f"[WARN] GraphRAG search failed, falling back to keyword search: {e}")
                use_vector_search = False

        # === Keyword-based Search (original logic) ===
        if not use_vector_search:
            # Build query entities: food items + conditions + restrictions + default entities
            all_entities = []
            keywords = get_keywords(meal_plan_names)
            all_entities.extend(keywords)
            all_entities.extend(conditions + conditions + restrictions + list(DIETARY_QUERY_ENTITIES))

            # Remove duplicates while preserving order
            all_entities = list(dict.fromkeys(all_entities))

            # Query KG for each entity, filtering by prioritized risk relations
            for entity in all_entities[:15]:  # Limit to 15 entities for performance
                try:
                    search_results = kg.search_entities(entity)

                    for result in search_results:
                        entity_name = result.get("head", "")
                        tail = result.get("tail", "")
                        rel_type = result.get("rel_type", "")
                        condition = result.get("condition", "")

                        # Filter by prioritized risk relations
                        # if rel_type not in prioritized_risk_kg_rels:
                        #     continue

                        if not tail:
                            continue

                        results.append({
                            "entity": entity_name,
                            "relation": rel_type,
                            "related_to": tail,
                            "condition": condition
                        })
                except Exception as e:
                    print(f"[WARN] Failed to query entity {entity}: {e}")

        # Format results for prompt
        if not results:
            return "No relevant KG data found."

        # context_lines = ["## Relevant Knowledge Graph Relationships"]
        context_lines = []
        # Deduplicate results
        seen_relations = set()
        unique_results = []
        for r in results:
            key = f"{r['entity']}-{r['relation']}-{r['related_to']}"
            if key not in seen_relations:
                seen_relations.add(key)
                unique_results.append(r)

        for r in unique_results[:20]:  # Limit to 20 most relevant results
            # context_lines.append(f"- {r['entity']} --[{r['relation']}]--> {r['related_to']}")
            context_lines.append("<{}, {}, {}> regarding {}".format(r['entity'], r['relation'], r['related_to'], r['condition']))

        return "\n".join(context_lines)

    def _query_exercise_kg_for_assessment(
        self,
        plan: Dict[str, Any],
        user_metadata: Dict[str, Any],
        use_vector_search: bool = True  # GraphRAG: use vector search instead of keyword matching
    ) -> str:
        """
        Query knowledge graph for exercise plan safety assessment using GraphRAG approach

        Args:
            plan: Exercise plan with exercises
            user_metadata: User metadata including conditions
            use_vector_search: If True, use vector search (GraphRAG); else use keyword matching
        """
        kg = get_kg_query()
        results = []

        # Extract exercise names from plan
        exercise_names = []
        sessions = plan.get("sessions", {})
        if isinstance(sessions, dict):
            for session_key, session in sessions.items():
                if isinstance(session, dict):
                    exercises = session.get("exercises", [])
                    for ex in exercises:
                        if isinstance(ex, dict):
                            ex_name = ex.get("name", "")
                            if ex_name:
                                exercise_names.append(ex_name)
                elif hasattr(session, 'exercises'):
                    for ex in session.exercises:
                        if hasattr(ex, 'name'):
                            exercise_names.append(ex.name)
        
        # print(f"[DEBUG] assessor judging exercise names = {exercise_names}")

        # Get user conditions
        conditions = user_metadata.get("medical_conditions", [])

        # === GraphRAG Approach: Vector Search + Graph Traversal ===
        use_vector_search = False
        if use_vector_search:
            try:
                # 1. For each exercise, use vector search to find similar entities
                seen_entities = set()
                for exercise_name in exercise_names[:5]:  # Limit to 5 exercises
                    anchors = kg.search_similar_entities(exercise_name, top_k=2)

                    for anchor in anchors:
                        anchor_name = anchor.get("name", "")
                        if not anchor_name:
                            continue

                        if anchor_name not in seen_entities:
                            seen_entities.add(anchor_name)
                            # Get neighbors for graph traversal (1-hop)
                            neighbors = kg.client.get_neighbors(anchor_name)

                            for neighbor in neighbors:
                                entity_name = neighbor.get("neighbor", "")
                                rel_type = neighbor.get("rel_type", "")

                                if not entity_name:
                                    continue

                                # Filter by prioritized exercise risk relations
                                # All relation types are accepted for exercise (None filter)
                                results.append({
                                    "entity": anchor_name,
                                    "relation": rel_type,
                                    "related_to": entity_name
                                })

                # 2. Also add conditions
                for condition in conditions:
                    anchors = kg.search_similar_entities(condition, top_k=2)
                    for anchor in anchors:
                        anchor_name = anchor.get("name", "")
                        if not anchor_name:
                            continue

                        neighbors = kg.client.get_neighbors(anchor_name)
                        for neighbor in neighbors:
                            entity_name = neighbor.get("neighbor", "")
                            rel_type = neighbor.get("rel_type", "")

                            if not entity_name:
                                continue

                            results.append({
                                "entity": anchor_name,
                                "relation": rel_type,
                                "related_to": entity_name
                            })

            except Exception as e:
                print(f"[WARN] GraphRAG search failed, falling back to keyword search: {e}")
                use_vector_search = False

        # === Fallback: Keyword-based Search (original logic) ===
        if not use_vector_search:
            all_entities = []
            for entity_list in [exercise_names, conditions]:
                for entity in entity_list:
                    keywords = get_keywords(entity)
                    all_entities.extend(keywords)

            # Remove duplicates while preserving order
            all_entities = list(dict.fromkeys(all_entities))

            # Exercise-specific risk relations (None = accept all relations)
            exercise_risk_rels = None

            # Query KG for each entity
            for entity in all_entities[:15]:
                try:
                    search_results = kg.search_entities(entity)

                    for result in search_results:
                        entity_name = result.get("head", "")
                        tail = result.get("tail", "")
                        rel_type = result.get("rel_type", "")

                        # Filter by exercise risk relations
                        if exercise_risk_rels is not None and rel_type not in exercise_risk_rels:
                            continue

                        if not tail:
                            continue

                        results.append({
                            "entity": entity_name,
                            "relation": rel_type,
                            "related_to": tail
                        })
                except Exception as e:
                    print(f"[WARN] Failed to query entity {entity}: {e}")

        # Format results for prompt
        if not results:
            return "No relevant KG data found."

        # context_lines = ["## Relevant Knowledge Graph Relationships"]
        context_lines = []
        # Deduplicate results
        seen_relations = set()
        unique_results = []
        for r in results:
            key = f"{r['entity']}-{r['relation']}-{r['related_to']}"
            if key not in seen_relations:
                seen_relations.add(key)
                unique_results.append(r)

        for r in unique_results[:20]:
            context_lines.append(f"- {r['entity']} --[{r['relation']}]--> {r['related_to']}")

        return "\n".join(context_lines)

    def _generate_recommendations(
        self,
        risk_factors: List[RiskFactor],
        plan_type: str,
        user_metadata: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        for rf in risk_factors:
            recommendations.append(rf.recommendation)

        # Add general recommendations
        if user_metadata.get("medical_conditions"):
            recommendations.append(
                f"Consult healthcare provider before starting due to: {', '.join(user_metadata['medical_conditions'])}"
            )

        if plan_type == "exercise":
            recommendations.extend([
                "Start gradually and listen to your body",
                "Stay hydrated before, during, and after exercise",
                "Stop immediately if you experience pain or discomfort"
            ])

        return list(set(recommendations))  # Remove duplicates


def assess_plan_safety(
    plan: Dict[str, Any],
    plan_type: str,
    user_metadata: Dict[str, Any],
    environment: Dict[str, Any] = {}
) -> SafetyAssessment:
    agent = SafeguardAgent()
    return agent.assess(plan, plan_type, user_metadata, environment)


def combined_assessment(
    diet_plan: Optional[Dict[str, Any]],
    exercise_plan: Optional[Dict[str, Any]],
    user_metadata: Dict[str, Any],
    environment: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Assess both diet and exercise plans together.
    Returns combined safety evaluation.
    """
    agent = SafeguardAgent()

    diet_assessment = None
    exercise_assessment = None

    if diet_plan:
        diet_assessment = agent.assess(diet_plan, "diet", user_metadata, environment)

    if exercise_plan:
        exercise_assessment = agent.assess(exercise_plan, "exercise", user_metadata, environment)

    # Calculate combined score
    assessments = [a for a in [diet_assessment, exercise_assessment] if a]
    if assessments:
        overall_score = sum(a.score for a in assessments) // len(assessments)
        is_safe = all(a.is_safe for a in assessments)
    else:
        overall_score = 100
        is_safe = True

    return {
        "diet_assessment": diet_assessment,
        "exercise_assessment": exercise_assessment,
        "overall_score": overall_score,
        "is_safe": is_safe
    }


if __name__ == "__main__":
    diet_plan = {
        "id": 1,
        "meal_plan": {
            "breakfast": [
                {"food": "Oatmeal", "portion": "1bowl", "calories": 150, "protein": 5, "carbs": 27, "fat": 3},
                {"food": "Milk", "portion": "200ml", "calories": 120, "protein": 8, "carbs": 12, "fat": 5}
            ],
            "lunch": [
                {"food": "Rice", "portion": "100g", "calories": 130, "protein": 3, "carbs": 28, "fat": 0},
                {"food": "Chicken", "portion": "100g", "calories": 165, "protein": 31, "carbs": 0, "fat": 4}
            ],
            "dinner": []
        },
        "total_calories": 1500,
        "calories_deviation": -8.5,
        "macro_nutrients": {
            "protein": 120,
            "carbs": 200,
            "fat": 45,
            "protein_ratio": 0.32,
            "carbs_ratio": 0.53,
            "fat_ratio": 0.27
        },
        "safety_notes": []
    }

    # Exercise plan with ExercisePlan-like structure
    exercise_plan = {
        "id": 1,
        "title": "Weekly Cardio Plan",
        "sessions": {
            "morning": {
                "time_of_day": "morning",
                "exercises": [
                    {"name": "Walking", "exercise_type": "cardio", "duration_minutes": 30, "intensity": "low", "calories_burned": 150}
                ],
                "total_duration_minutes": 30,
                "total_calories_burned": 150,
                "overall_intensity": "low"
            },
            "afternoon": {
                "time_of_day": "afternoon",
                "exercises": [
                    {"name": "HIIT", "exercise_type": "hiit", "duration_minutes": 20, "intensity": "high", "calories_burned": 250}
                ],
                "total_duration_minutes": 20,
                "total_calories_burned": 250,
                "overall_intensity": "high"
            }
        },
        "total_duration_minutes": 50,
        "total_calories_burned": 400,
        "weekly_frequency": 5,
        "progression": "Increase duration by 5 mins each week",
        "safety_notes": []
    }

    user_metadata = {
        "age": 35,
        "gender": "male",
        "height_cm": 175,
        "weight_kg": 70,
        "medical_conditions": ["diabetes"],
        "fitness_level": "intermediate"
    }

    environment = {
        "weather": {"condition": "clear", "temperature_c": 25},
        "time_context": {"season": "summer"}
    }

    print("=== Testing Diet Plan Assessment ===")
    diet_assessment = assess_plan_safety(
        plan=diet_plan,
        plan_type="diet",
        user_metadata=user_metadata,
        environment=environment
    )
    print(f"Score: {diet_assessment.score}/100")
    print(f"Is Safe: {diet_assessment.is_safe}")
    print(f"Risk Level: {diet_assessment.risk_level.value}")
    print(f"Risk Factors: {len(diet_assessment.risk_factors)}")
    for rf in diet_assessment.risk_factors:
        print(f"  - {rf.factor}: {rf.description}")

    print("\n=== Testing Exercise Plan Assessment ===")
    exercise_assessment = assess_plan_safety(
        plan=exercise_plan,
        plan_type="exercise",
        user_metadata=user_metadata,
        environment=environment
    )
    print(f"Score: {exercise_assessment.score}/100")
    print(f"Is Safe: {exercise_assessment.is_safe}")
    print(f"Risk Level: {exercise_assessment.risk_level.value}")
    print(f"Risk Factors: {len(exercise_assessment.risk_factors)}")
    for rf in exercise_assessment.risk_factors:
        print(f"  - {rf.factor}: {rf.description}")

    print("\n=== Testing Diabetes with High Sugar Foods ===")
    bad_diet_plan = {
        "id": 2,
        "meal_plan": {
            "breakfast": [
                {"food": "Chocolate Cake", "portion": "1piece", "calories": 500, "protein": 5, "carbs": 60, "fat": 25},
                {"food": "Ice Cream", "portion": "1bowl", "calories": 300, "protein": 5, "carbs": 40, "fat": 15}
            ]
        },
        "total_calories": 800,
        "calories_deviation": -46,
        "macro_nutrients": {
            "protein": 10,
            "carbs": 100,
            "fat": 40,
            "protein_ratio": 0.05,
            "carbs_ratio": 0.50,
            "fat_ratio": 0.45
        },
        "safety_notes": []
    }
    diabetes_assessment = assess_plan_safety(
        plan=bad_diet_plan,
        plan_type="diet",
        user_metadata=user_metadata,
        environment=environment
    )
    print(f"Score: {diabetes_assessment.score}/100")
    print(f"Is Safe: {diabetes_assessment.is_safe}")
    print(f"Risk Factors: {len(diabetes_assessment.risk_factors)}")
    for rf in diabetes_assessment.risk_factors:
        print(f"  - {rf.factor}: {rf.description}")

    print("\n=== All Tests Passed ===")
