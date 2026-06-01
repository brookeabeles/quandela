"""Compatibility entrypoint. Canonical: ``phasecraft.experiments.lr_scaling.sweep_lr_depth_until_win``."""
from phasecraft.experiments.lr_scaling.sweep_lr_depth_until_win import *  # noqa: F401,F403

if __name__ == "__main__":
    from phasecraft.experiments.lr_scaling.sweep_lr_depth_until_win import main

    main()
