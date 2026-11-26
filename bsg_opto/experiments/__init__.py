"""Experiment registry for bsg_opto."""
from . import (
    random_exc,
    random_vs_highdrive,
    ei_connectivity,
    tonic_drive,
    tonic_varlight,
    opsin_to_i_strength,
    state_dependence,
    tonic_altconn,
)

EXPERIMENTS = {
    'random-exc': random_exc,
    'random-vs-highdrive': random_vs_highdrive,
    'ei-connectivity': ei_connectivity,
    'tonic-drive': tonic_drive,
    'tonic-varlight': tonic_varlight,
    'opsin-to-i-strength': opsin_to_i_strength,
    'state-dependence': state_dependence,
    'tonic-altconn': tonic_altconn,
}
