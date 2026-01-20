import re
from typing import Optional


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Best-effort extraction of Python code from a model generation."""
    if not text:
        return ""

    # Prefer fenced code blocks.
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    # Fallback: strip common leading chatter and keep from first `def`/`class`.
    lowered = text.lower()
    def_idx = lowered.find("def ")
    class_idx = lowered.find("class ")
    candidates = [i for i in [def_idx, class_idx] if i != -1]
    if candidates:
        start = min(candidates)
        return text[start:].strip()

    return text.strip()
