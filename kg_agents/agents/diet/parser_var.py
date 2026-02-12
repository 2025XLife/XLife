from typing import List, Dict, Any
from .models import BaseFoodItem

class DietPlanParser:
    def __init__(self, num_variants: int = 3, min_scale: float = 0.5, max_scale: float = 1.5):
        # Generate variant configurations uniformly distributed between min_scale and max_scale
        self.num_variants = num_variants
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Generate scale factors uniformly distributed between min_scale and max_scale
        if num_variants == 1:
            scale_factors = [(min_scale + max_scale) / 2]
        elif num_variants == 2:
            scale_factors = [min_scale, max_scale]
        else:
            step = (max_scale - min_scale) / (num_variants - 1)
            scale_factors = [min_scale + i * step for i in range(num_variants)]

        # Generate variant names: Variant_1, Variant_2, etc.
        self.variant_configs = [
            (f"Variant_{i+1}", round(factor, 3))
            for i, factor in enumerate(scale_factors)
        ]
        self.variants = {name: factor for name, factor in self.variant_configs}
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
            variants = [name for name, _ in self.variant_configs]
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
        # Continuous units: direct multiplication
        if unit in ["gram", "ml"]:
            return round(original * scale_factor, 1)
        # Discrete units: round to nearest increment
        elif unit in ["piece", "slice", "cup", "bowl"]:
            increment = self.unit_adjustments.get(unit, 0.5)
            target = original * scale_factor
            # Round to nearest increment
            if increment > 0:
                rounded = round(target / increment) * increment
                # Apply minimum value
                min_value = increment
                return round(max(min_value, rounded), 1)
            else:
                return round(target, 1)
        # Spoon: no adjustment (keep exact)
        elif unit == "spoon":
            return original
        # Unknown units: treat as continuous
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
        Dict with Variant_1, Variant_2, ... variants (default 3)
    """
    parser = DietPlanParser()
    return parser.expand_plan(base_items)