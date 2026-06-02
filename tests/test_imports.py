def test_core_imports():
    import phasecraft
    import phasecraft.lib.sim.bm24_qaoa_sim
    import phasecraft.experiments.lr_scaling
    import phasecraft.experiments.bm24_saddle_audit_p1
    import phasecraft.experiments.w_saddle


def test_legacy_entrypoint_imports():
    import phasecraft.bm24_qaoa_sim
    import phasecraft.train_lr_notebook_protocol
