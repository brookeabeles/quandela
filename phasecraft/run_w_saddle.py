"""Compatibility re-export. Canonical: ``phasecraft.shims.run_w_saddle``."""
from phasecraft.shims.run_w_saddle import *  # noqa: F401,F403

if __name__ == "__main__":
    from phasecraft.shims.run_w_saddle import main

    main()
