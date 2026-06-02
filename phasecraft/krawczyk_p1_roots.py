"""Compatibility re-export. Canonical: ``phasecraft.lib.saddles.krawczyk_p1_roots``."""
from phasecraft.lib.saddles.krawczyk_p1_roots import *  # noqa: F401,F403
import phasecraft.lib.saddles.krawczyk_p1_roots as _impl

# Keep legacy private-symbol imports working for older audit scripts.
_check_krawczyk_at_uniform_radius = _impl._check_krawczyk_at_uniform_radius
