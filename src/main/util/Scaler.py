import re
from typing import List, Callable


class Scaler:
    @classmethod
    def to_unit(cls, value: str | int | float, unit: str) -> (float, str):
        """
        Normalize value by scaling it according to the unit modifier
        and remove the scale term from the unit.

        Args:
            value: the value
            unit: Unit string that may contain scale terms

        Returns:
            tuple: (normalized_value, cleaned_unit)
        """

        # Dictionary of scale factors
        scale_factors = {
            'thousand': 1000,
            'k': 1000,
            'million': 1000_000,
            'm': 1000_000,
            'billion': 1000_000_000,
            'b': 1000_000_000,
            'trillion': 1000_000_000_000,
            't': 1000_000_000_000
        }

        # Convert value to float
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid value format: {value}")

        # Check if any scale term is in the unit
        scale_factor = 1
        cleaned_unit = unit
        unit_words = unit.split()
        for i, w in enumerate(unit_words):
            for scale_term in scale_factors:
                if scale_term == w.lower().strip():
                    # the scale term is in the unit
                    scale_factor = scale_factors[scale_term]
                    cleaned_unit = " ".join(unit_words[:i] + unit_words[i+1:])
                    break

        # Scale the value
        normalized_value = value * scale_factor

        return normalized_value, cleaned_unit

    @classmethod
    def to_magnitude(cls, value: str | int | float) -> (float, str):
        """
        Scale the value to a proper magnitude.  Returns the scaled value and its magnitude.

        Args:
            value: the value

        Returns:
            tuple: (scaled_value, unit_modifier)
        """
        # Normalize to single unit first
        amount, _ = cls.to_unit(value, "")

        # Define scale thresholds
        scales = {
            1_000_000_000_000: "trillion",
            1_000_000_000: "billion",
            1_000_000: "million",
            1_000: "thousand"
        }

        # Handle sign
        sign = -1 if amount < 0 else 1
        amount = abs(amount)

        # Find the appropriate scale
        for scale, name in scales.items():
            if amount >= scale:
                return sign * amount / scale, name

        return sign * amount, ""

    DEFAULT_CURRENCY_CODES = ["USD", "EUR", "JPY", "NTD"]

    @classmethod
    def normalize_currency(cls, value: str | int | float, unit: str | None, **kwargs) -> (str, str):
        if not unit:
            return None, None

        more_codes = kwargs.get("more_codes", [])

        # Check if this is a currency amount.  Return None if not.
        if isinstance(more_codes, str):
            more_codes = [more_codes]

        currency_codes = cls.DEFAULT_CURRENCY_CODES + more_codes

        unit_words = [w.upper() for w in unit.split()]
        if all([cc not in unit_words for cc in currency_codes]):
            return None, None

        # Convert to single unit
        try:
            value, unit = cls.to_unit(value, unit)
        except ValueError:
            return None, None

        # Adjust to magnitude
        value, modifier = cls.to_magnitude(value)

        if modifier:
            # There is a modifier.  Conventionally, the value shall be formatted without trailing 0's
            return f"{value:.2f}".rstrip('0').rstrip('.'), f"{modifier} {unit}"
        else:
            # There is no modifier.  Conventionally, the value shall be formatted with to decimal points
            return f"{value:.2f}", unit

    @classmethod
    def normalize_count(cls, value: str | int | float, unit: str | None) -> (str, str):
        if unit is None:
            unit = ""

        # Convert to single unit
        try:
            value, unit = cls.to_unit(value, unit)
        except ValueError:
            return None, None

        # Adjust to magnitude
        value, modifier = cls.to_magnitude(value)

        # If unit is each, no need to specify if there is a unit modifier
        if modifier and (unit == "ea" or unit == "each"):
            unit = ""

        return f"{value:.2f}".rstrip('0').rstrip('.'), f"{modifier} {unit}".strip()

    @classmethod
    def normalize_multiple(cls, value: str | int | float, unit: str | None) -> (str, str):
        if unit in ["x", "X"]:
            unit = "times"

        if isinstance(value, str):
            match = re.match(r"(\d+(?:\.\d+)?)\s*[xX]", value)
            if match:
                if not unit or unit in ["x", "X", "times"]:
                    value = match.group(1)
                    unit = "times"

        if unit in ["x", "X"]:
            unit = "times"
        if unit != "times":
            return None, None

        try:
            value, modifier = cls.to_magnitude(value)
        except ValueError:
            return None, None

        return f"{value:.2f}".rstrip('0').rstrip('.'), f"{modifier} {unit}".strip()

    @classmethod
    def normalize(cls, value: str | int | float, unit: str | None, **kwargs) -> (float, str):
        functions: List[Callable] = [
            cls.normalize_currency,
            cls.normalize_multiple,
            cls.normalize_count,
        ]

        for f in functions:
            v, u = f(value, unit, **kwargs)
            if v:
                return v, u

        return value, unit



