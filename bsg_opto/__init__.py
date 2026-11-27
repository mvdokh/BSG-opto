"""
bsg_opto: Optogenetic stimulation experiments on the Lindén et al. (2022)
balanced spinal network model (BergLab/BSG).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSG_DIR = os.path.join(ROOT, 'BSG')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, '.mplcache'))
if BSG_DIR not in sys.path:
    sys.path.insert(0, BSG_DIR)

from .config import NetworkConfig, NeuronConfig, ProtocolConfig, ChR2Config, RESULTS_DIR
from .network import BalancedNetwork
from .licking import LickingNetworkConfig, LickingNetwork, licking_neuron_config, generate_W_licking

__all__ = [
    'ROOT', 'BSG_DIR', 'RESULTS_DIR',
    'NetworkConfig', 'NeuronConfig', 'ProtocolConfig', 'ChR2Config',
    'BalancedNetwork',
    'LickingNetworkConfig', 'LickingNetwork', 'licking_neuron_config', 'generate_W_licking',
]
