"""BM24 p=1 saddle audit pipeline (z-chart Krawczyk vs Prop-4 finite-n)."""

__all__ = ["run_audit"]


def run_audit(*args, **kwargs):
    from .audit import run_audit as _run

    return _run(*args, **kwargs)
