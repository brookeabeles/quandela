"""Shim → ``phasecraft.w_saddle.workflow`` + ``resolve`` CLI."""

from phasecraft.w_saddle.workflow import *  # noqa: F401,F403


def main() -> None:
    import sys

    from phasecraft.w_saddle.cli import main as run_main

    argv = list(sys.argv[1:])
    mapped: list[str] = ["resolve"]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--sweep-in":
            mapped.extend(argv[i : i + 2])
            i += 2
            continue
        if a == "--in" and i + 1 < len(argv):
            mapped.extend(["--sweep-in", argv[i + 1]])
            i += 2
            continue
        mapped.append(a)
        i += 1
    run_main(mapped)


if __name__ == "__main__":
    main()
