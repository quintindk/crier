"""CPU model detection helpers used by NPU-class backends.

We use these to give earlier, friendlier diagnostics: telling a user that
their NPU is the wrong XDNA generation is much more useful than letting
the import succeed and then watching VitisAI EP fail to load.

Detection is intentionally name-based and conservative: when we cannot
classify a CPU we return ``None`` rather than guessing.
"""

from __future__ import annotations

import platform
import re
import subprocess
from typing import Literal

XdnaGen = Literal["xdna1", "xdna2"]

# Phoenix (Ryzen 7040) and Hawk Point (Ryzen 8040) ship XDNA 1.
# Pattern catches: "Ryzen 7 7840HS", "Ryzen 7 PRO 7840HS", "Ryzen 5 8645H",
# "Ryzen 9 8945HX", "Ryzen 7 PRO 8845HS" and friends.
_XDNA1_PATTERN = re.compile(
    r"\bRyzen\s+[3579]\s+(?:PRO\s+)?[78][2-9]\d{2}[A-Z]*\b",
    re.IGNORECASE,
)

# Strix Point / Strix Halo / Krackan Point all use the "Ryzen AI" brand
# prefix, which AMD reserved for XDNA 2-class silicon.
_XDNA2_PATTERN = re.compile(r"\bRyzen\s+AI\b", re.IGNORECASE)


def _read_cpu_model_name() -> str | None:
    """Best-effort lookup of the human CPU model string. Returns None on failure."""
    system = platform.system()
    try:
        if system == "Windows":
            out = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in out.stdout.splitlines():
                if line.startswith("Name="):
                    return line.split("=", 1)[1].strip() or None
            return None
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
            return None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def detect_xdna_generation(model_name: str | None = None) -> XdnaGen | None:
    """Classify the host CPU's NPU generation, or return None if unsure.

    Pass ``model_name`` explicitly in tests to bypass platform probing.
    """
    name = model_name if model_name is not None else _read_cpu_model_name()
    if not name:
        return None
    if _XDNA2_PATTERN.search(name):
        return "xdna2"
    if _XDNA1_PATTERN.search(name):
        return "xdna1"
    return None
