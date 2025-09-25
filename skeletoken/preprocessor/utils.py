from tokenizers import Regex


def replace_pattern(obj: dict) -> dict:
    """Replaces a 'pattern' field in a dictionary with the appropriate Regex or string."""
    if "pattern" in obj and obj["pattern"] is not None:
        if "String" in obj["pattern"]:
            obj["pattern"] = obj["pattern"]["String"]
        elif "Regex" in obj["pattern"]:
            obj["pattern"] = Regex(obj["pattern"]["Regex"])
        else:
            raise ValueError(f"Unknown pattern type {obj['pattern']}")

    return obj
