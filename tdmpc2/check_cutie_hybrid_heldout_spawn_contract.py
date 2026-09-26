"""Regression contract for held-out evaluator imports under multiprocessing spawn."""

from __future__ import annotations

import importlib
import importlib.util
import multiprocessing as mp
from pathlib import Path
import traceback


MODULE = "tdmpc2.tools.evaluate_cutie_hybrid_heldout"


def _spawn_probe(connection) -> None:
    try:
        package = importlib.import_module("tdmpc2")
        module = importlib.import_module(MODULE)
        if not hasattr(package, "__path__"):
            raise AssertionError("tdmpc2 resolved to a module instead of the package")
        project_dir = Path(module.__file__).resolve().parents[1]
        repo_dir = project_dir.parent
        if [Path(value).resolve() for value in __import__("sys").path[:2]] != [
            repo_dir,
            project_dir,
        ]:
            raise AssertionError("local repository import roots are not canonical")
        origins = {}
        for name in ("tdmpc2", "common", "envs", "perception"):
            spec = importlib.util.find_spec(name)
            if spec is None or spec.origin is None:
                raise AssertionError(f"cannot resolve local module {name}")
            origin = Path(spec.origin).resolve()
            if project_dir not in origin.parents and origin != project_dir:
                raise AssertionError(f"{name} resolved outside PROJECT_DIR: {origin}")
            origins[name] = str(origin)
        connection.send(
            {
                "ok": True,
                "module": module.__name__,
                "package_paths": list(package.__path__),
                "local_origins": origins,
            }
        )
    except BaseException:
        connection.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        connection.close()


def main() -> None:
    # Importing the evaluator mutates sys.path exactly as it will in production.
    importlib.import_module(MODULE)
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_spawn_probe, args=(child,))
    process.start()
    child.close()
    try:
        if not parent.poll(30.0):
            process.terminate()
            raise AssertionError("spawn import probe timed out")
        result = parent.recv()
    finally:
        parent.close()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
    if process.exitcode != 0:
        raise AssertionError(f"spawn import probe exited with {process.exitcode}")
    if result.get("ok") is not True or result.get("module") != MODULE:
        raise AssertionError(result)
    print("CUTIE_HYBRID_HELDOUT_SPAWN_CONTRACT_OK", result)


if __name__ == "__main__":
    main()
