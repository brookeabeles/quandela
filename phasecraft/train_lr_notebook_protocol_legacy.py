"""Compatibility entrypoint. Canonical: ``phasecraft.experiments.lr_scaling.train_lr_notebook_protocol_legacy``."""
from phasecraft.experiments.lr_scaling.train_lr_notebook_protocol_legacy import *  # noqa: F401,F403

if __name__ == "__main__":
    from phasecraft.experiments.lr_scaling.train_lr_notebook_protocol_legacy import main

    main()
