# BSG-opto

Optogenetic stimulation experiments built on the **Lindén et al. (2022)** balanced spinal motor network model ([BergLab/BSG](https://github.com/BergLab/BSG), *Nature* 610:526–531).

The upstream repository is kept untouched in `BSG/`. All reusable simulation, optogenetics, connectivity, analysis, and plotting helpers live in `bsg_opto/`, and every experiment can be run from a single entry point (`run.py`).

## Repository Structure

```text
BSG-opto/
├── run.py                  # Unified CLI to list and run any experiment
├── requirements.txt        # Python dependencies
├── BSG/                    # Upstream BergLab/BSG repo (untouched)
├── bsg_opto/               # Modular helper package & experiment definitions
│   ├── config.py           # Network, neuron, protocol, and ChR2 parameter dataclasses
│   ├── network.py          # BalancedNetwork builder, eigenmodes, phase & E/I phase sorting
│   ├── opsin.py            # Step, fixed-desens, and intensity-dependent ChR2 models; cell selection
│   ├── connectivity.py     # Rewiring & weight scaling (opsin->I prob/weight, opsin->all hubs)
│   ├── simulate.py         # Single-trial (spinalsim) and batched BLAS simulators
│   ├── analysis.py         # Linear stability (Re λ), rhythmicity metrics, PCA, nerve readouts (limb / XII)
│   ├── plotting.py         # Shared raster, mean-rate, nerve, light-model, and heatmap helpers
│   ├── cli.py              # CLI registry and argument dispatch
│   └── experiments/        # Experiment modules (each exposes add_args + run)
│       ├── random_exc.py
│       ├── random_vs_highdrive.py
│       ├── ei_connectivity.py
│       ├── tonic_drive.py
│       ├── tonic_varlight.py
│       ├── opsin_to_i_strength.py
│       ├── state_dependence.py
│       └── tonic_altconn.py
└── results/                # Consolidated figures & summary tables for all experiments
    ├── 01_random_exc/
    ├── 02_random_vs_highdrive/
    ├── 03_ei_connectivity/
    ├── 04_tonic_drive_5pct/
    ├── 05_tonic_varlight_10pct/
    ├── 06_tonic_varlight_N2000/
    ├── 07_opsin_to_i_strength/
    ├── 08_state_dependence/
    └── 09_tonic_altconn/
        ├── I10/
        ├── I20/
        ├── I40/
        ├── I80/
        ├── I160/
        └── I320/
```

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running Experiments

List all available experiments:

```bash
.venv/bin/python run.py list
```

Inspect options for any experiment:

```bash
.venv/bin/python run.py <experiment> --help
```

Examples:

```bash
# 1. Random E-cell photostimulation from rest (15% or 5% of E cells)
.venv/bin/python run.py random-exc --pct 15
.venv/bin/python run.py random-exc --pct 5

# 2. Random vs high-drive / most-active / greedy model-best E cells
.venv/bin/python run.py random-vs-highdrive --pct 15
.venv/bin/python run.py random-vs-highdrive --pct 5

# 3. Higher opsin E -> I connection probability (rewire vs add)
.venv/bin/python run.py ei-connectivity --pct 5

# 4. Uniform tonic drive (D = 1..15) + fixed-light desensitizing ChR2 in 5% of E
.venv/bin/python run.py tonic-drive --pct 5 --d-min 1 --d-max 15

# 5. Tonic drive (D = 0..20) + random light per cell & intensity-dependent desensitization
.venv/bin/python run.py tonic-varlight --pct 10 --N 200 --d-min 0 --d-max 20
.venv/bin/python run.py tonic-varlight --pct 10 --N 2000 --d-min 1 --d-max 15

# 6. Stronger opsin E -> I synaptic weights (x1..x16) across tonic drive
.venv/bin/python run.py opsin-to-i-strength --N 200

# 7. State dependence (quiescent vs aroused mu, baseline spread sigma) + sweep figure
.venv/bin/python run.py state-dependence
.venv/bin/python run.py state-dependence --sweep-only

# 8. Opsin -> all cells hub connectivity across one or more max light intensities
.venv/bin/python run.py tonic-altconn --i-max 80
.venv/bin/python run.py tonic-altconn --i-max 10 20 40 80 160 320
```

## Experiment Catalog & Summary of Findings

| # | CLI Command | Output Folder | Question Tested | Main Finding |
|---|---|---|---|---|
| 1 | `random-exc` | `results/01_random_exc/` | Does lighting 15% or 5% of E cells from rest start the rhythm? | No. Lit cells fire tonically; `Re λ` stays below 1 across all intensities and fractions. |
| 2 | `random-vs-highdrive` | `results/02_random_vs_highdrive/` | Does lighting the E cells carrying the reference rhythm help? | No; equal to or worse than random. Pinning high-drive cells near saturation removes them from the loop. |
| 3 | `ei-connectivity` | `results/03_ei_connectivity/` | Does higher opsin E→I connection probability evoke a rhythm? | No; extra inhibition suppresses unlit E cells and stabilizes the network further. |
| 4 | `tonic-drive` | `results/04_tonic_drive_5pct/` | Does 5% E ChR2 work on top of subthreshold tonic drive (`D = 1..15`)? | Only near threshold (`D ≈ 11–13`, pick-dependent). Above threshold (`D ≥ 14`), light weakens the rhythm. |
| 5 | `tonic-varlight` | `results/05_tonic_varlight_10pct/` | 10% E with random light per cell (`φ ~ U(0,1)`) + intensity-dependent desensitization (`D = 0..20`) + XII nerve readout | Uneven light beats uniform full light near threshold (`D = 9–13`) and is less disruptive above it (`D = 14–20`). |
| 6 | `tonic-varlight --N 2000` | `results/06_tonic_varlight_N2000/` | Same as #5 with `N = 2000` (1000 E : 1000 I, `C = 0.1`) | Stronger response even at `D = 1` (partial rhythm), largely because `C·N / √N = √N` scaling amplifies E-only imbalance. |
| 7 | `opsin-to-i-strength` | `results/07_opsin_to_i_strength/` | Stronger opsin E→I synaptic weights (`×2` to `×16`) across `D = 1..15` | Suppresses both unlit E firing and rhythmicity at every drive level. |
| 8 | `state-dependence` | `results/08_state_dependence/` | Quiescent vs aroused baseline (`μ = 0..13`) and per-cell spread (`σ = 0..6`) | Strong state dependence for uniform drive (0% → 55% success), largely erased by uneven baseline drive (`σ = 4–6`) or `×4` opsin→I weights. |
| 9 | `tonic-altconn` | `results/09_tonic_altconn/I<val>/` | Opsin E→all-cells hub rewiring (fixed in-degree) across `p = 0.1..1.0` and max light `I = 10, 20, 40, 80, 160, 320` | Hub wiring raises low-`D` activity, but stronger light or high `p` clamps excitatory input and suppresses rhythms at high `D`. |
