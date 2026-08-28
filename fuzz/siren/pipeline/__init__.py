"""SIREN evaluation pipeline: find -> verify -> fuzz -> measure.

See README.md in this directory. Stage entry points:

    stage1_search    find ground-truth attacks in the SPARK benchmark
    stage2_verify    re-verify from scratch, record the trace, render
    stage3_targets   package verified attacks as fuzz targets
    stage3_fuzz      run SIREN against a target, logging every evaluation
    stage4_curves    discovery curves: attacks vs trials, attacks vs time
    stage4_render    3-D render of every searched location
"""
