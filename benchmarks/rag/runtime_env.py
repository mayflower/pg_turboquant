from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence


def maybe_reexec_into_bergen_venv(
    *,
    script_path: str | Path,
    argv: Sequence[str] | None = None,
    required_modules: Sequence[str] = ("psycopg", "yaml", "torch", "datasets", "transformers"),
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    execvpe: Callable[[str, list[str], Mapping[str, str]], object] = os.execvpe,
    current_executable: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    argv = list(argv or sys.argv)
    if "--dry-run" in argv:
        return False

    script_path = Path(script_path).resolve()
    venv_python = script_path.parent / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return False

    executable = Path(current_executable or sys.executable).resolve()
    if executable == venv_python.resolve():
        return False

    if all(find_spec(module_name) is not None for module_name in required_modules):
        return False

    env = dict(environ or os.environ)
    execvpe(str(venv_python), [str(venv_python), *argv], env)
    return True
