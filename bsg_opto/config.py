"""Shared configuration dataclasses and default paths."""
from dataclasses import dataclass
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')


@dataclass(frozen=True)
class NetworkConfig:
    """Lindén et al. (2022) Fig. 2 balanced Dale network parameters."""
    N: int = 200
    C: float = 0.1
    frac_inh: float = 0.5
    g: float = 1.0
    W_radius: float = 1.0
    seed: int = 1

    @property
    def N_inh(self) -> int:
        return int(self.N * self.frac_inh)

    @property
    def N_exc(self) -> int:
        return self.N - self.N_inh

    def to_bsg_kwargs(self) -> dict:
        return dict(N=self.N, C=self.C, frac_inh=self.frac_inh,
                    g=self.g, W_radius=self.W_radius, seed=self.seed)


@dataclass(frozen=True)
class NeuronConfig:
    """Rate neuron transfer function and membrane dynamics parameters."""
    tau_V: float = 50.0
    noise_ampl: float = 4.0
    threshold: float = 20.0
    gain: float = 1.2
    fmax: float = 50.0
    V_init: float = 0.0


@dataclass(frozen=True)
class ProtocolConfig:
    """Stimulation timing (ms) and reference tonic drive."""
    T: int = 9000
    stim_on: int = 1000
    stim_off: int = 6000
    analysis_delay: int = 1000
    post_delay: int = 1000
    i_ref: float = 20.0

    @property
    def analysis_win(self) -> tuple[int, int]:
        return (self.stim_on + self.analysis_delay, self.stim_off)

    @property
    def post_win(self) -> tuple[int, int]:
        return (self.stim_off + self.post_delay, self.T)

    @property
    def pre_win(self) -> tuple[int, int]:
        return (max(0, self.stim_on - 500), self.stim_on)

    @property
    def tvec(self) -> np.ndarray:
        return np.arange(self.T) / 1000.0


@dataclass(frozen=True)
class ChR2Config:
    """Channelrhodopsin-2 photocurrent and desensitization parameters.

    At relative light phi = 1 (bright/saturating), steady/peak = ss_frac (0.3)
    and tau_des = 40 ms. As phi -> 0 (dim light), steady/peak -> ss_dim (0.9)
    and tau_des -> tau_des_dim (150 ms).
    """
    i_max: float = 80.0
    tau_des: float = 40.0
    ss_frac: float = 0.3
    tau_off: float = 15.0
    ss_dim: float = 0.9
    tau_des_dim: float = 150.0
    k_light: float = 0.25
    desensitize: bool = True
