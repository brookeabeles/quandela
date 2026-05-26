"""Shim → ``phasecraft.w_saddle.core`` (use ``python -m phasecraft.w_saddle certify``)."""

from phasecraft.w_saddle.core import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys

    from phasecraft.w_saddle.cli import main as run_main

    argv = list(sys.argv[1:])
    if not argv or (len(argv) == 1 and argv[0] == "--selftest"):
        from phasecraft.w_saddle.core import selftest

        selftest()
    elif "--selftest" in argv:
        from phasecraft.w_saddle.core import selftest

        selftest()
    else:
        run_main(["certify", *argv])
