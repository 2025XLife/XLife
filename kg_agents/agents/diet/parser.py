from typing import List, Dict, Any
from .models import BaseFoodItem


class DietPlanParser:
    def __init__(self):
        # Scaling factors for each variant
        self.variants = {
            "Lite": 0.8,       # 80% of base portion
            "Standard": 1.0,    # 100% of base portion
            "Plus": 1.2        # 120% of base portion
        }

        # Unit-specific adjustments for discrete quantities
        self.unit_adjustments = {
            # Continuous units: direct multiplication
            "gram": None,
            "ml": None,
            # Discrete units: half-unit steps
            "piece": 0.5,
            "slice": 1.0,
            "cup": 0.5,
            "bowl": 0.5,
            # Spoon: no adjustment (keep exact)
            "spoon": 0.0,
            # "serve": 0.0
        }

    def expand_plan(
        self,
        base_items: List[BaseFoodItem],
        variants: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        if variants is None:
            variants = ["Lite", "Standard", "Plus"]

        result = {}

        for variant_name in variants:
            scale_factor = self.variants.get(variant_name, 1.0)
            expanded_items = []

            for item in base_items:
                new_item = self._scale_item(item, scale_factor)
                new_item["_variant"] = variant_name
                expanded_items.append(new_item)

            result[variant_name] = expanded_items

        return result

    def _scale_item(
        self,
        item: BaseFoodItem,
        scale_factor: float
    ) -> Dict[str, Any]:
        unit = item.portion_unit
        original_num = item.portion_number

        if hasattr(item, 'total_calories') and item.total_calories is not None:
            original_total = item.total_calories
        elif hasattr(item, 'calories_per_unit') and item.calories_per_unit is not None:
            original_total = item.calories_per_unit * original_num
        else:
            original_total = 0

        calories_per_unit = original_total / original_num if original_num > 0 else 0

        scaled_num = self._calculate_scaled_number(original_num, unit, scale_factor)

        if original_num > 0:
            total_calories = round(original_total * (scaled_num / original_num), 1)
        else:
            total_calories = original_total

        return {
            "food_name": item.food_name,
            "portion_number": scaled_num,
            "portion_unit": unit,
            "calories_per_unit": round(calories_per_unit, 2),
            "total_calories": total_calories
        }

    def _calculate_scaled_number(
        self,
        original: float,
        unit: str,
        scale_factor: float
    ) -> float:
        if unit in ["gram", "ml"]:
            return round(original * scale_factor, 1)

        elif unit in ["piece", "slice", "cup", "bowl"]:
            adjustment = self.unit_adjustments.get(unit, 0.5)

            if scale_factor < 1.0:
                # Lite: reduce
                new_num = original - adjustment
                return max(0.5, round(new_num, 1))
            elif scale_factor > 1.0:
                # Plus: increase
                new_num = original + adjustment
                return round(new_num, 1)
            else:
                # Standard: keep original
                return original

        elif unit == "spoon":
            return original

        else:
            return round(original * scale_factor, 1)

    def expand_single_item(
        self,
        item: BaseFoodItem
    ) -> Dict[str, Dict[str, Any]]:
        return self.expand_plan([item])


# Convenience function
def expand_diet_plan(
    base_items: List[BaseFoodItem]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Quick helper to expand a diet plan.

    Args:
        base_items: List of BaseFoodItem from LLM

    Returns:
        Dict with "Lite", "Standard", "Plus" variants
    """
    parser = DietPlanParser()
    return parser.expand_plan(base_items)
