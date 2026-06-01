"""Shim → ``phasecraft.w_saddle.workflow`` + CLI."""

from phasecraft.w_saddle.workflow import *  # noqa: F401,F403


def main() -> None:
    import sys

    from phasecraft.w_saddle.cli import legacy_continue_argv, main as run_main

    run_main(legacy_continue_argv(sys.argv[1:]))


if __name__ == "__main__":
    main()
