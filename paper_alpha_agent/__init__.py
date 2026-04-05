"""Compatibility shim for running the src-layout package from the repo root.

This allows commands like:

```bash
python -m paper_alpha_agent.research.discovery --topic commodities --max-papers 5
```

to work even before the project has been installed in editable mode.

The real package lives under `src/paper_alpha_agent/`.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

_SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "paper_alpha_agent"
if _SRC_PACKAGE_DIR.exists():
    __path__.append(str(_SRC_PACKAGE_DIR))

__all__ = ["__version__"]
__version__ = "0.1.0"
