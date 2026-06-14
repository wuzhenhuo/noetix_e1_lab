# Template for Noetix-E1 Isaac Lab Projects

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RK](https://img.shields.io/badge/RSL_RL-3.0.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.
  This template requires Isaac Sim 5.0.0.（pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com）
  Isaac Lab 2.3.0 is required.(In most cases, simply follow the tutorial to install the latest version:“git clone git@github.com:isaac-sim/IsaacLab.git”)

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/NoetixE1    
    python -m pip install -e rsl_rl
- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/rsl_rl/train.py --task=<TASK_NAME>
        ```

## Ablation Experiments

This fork adds three incremental ablations on top of the PBHC baseline, each registered as an independent Gym environment.

### Modified / Added Files

| File | Change |
|------|--------|
| `source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/mdp/rewards.py` | Added `track_lin_vel_xy_adaptive_exp` function |
| `source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/__init__.py` | Registered 3 new environments |
| `source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/walkrun_cfg_contact_mask.py` | New env config: Contact Mask |
| `source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/walkrun_cfg_adaptive_tracking.py` | New env config: Adaptive Tracking |
| `source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/walkrun_cfg_curiosity_reward.py` | New env config: Curiosity Reward |
| `source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/agents/rsl_rl_amp_ppo_cfg_curiosity.py` | New agent config: RND enabled |

### Method Details

**PBHC Baseline** — `WalkRun-E1-v0` (original `walkrun_cfg.py`, unmodified)

**+ Contact Mask** — `WalkRun-E1-ContactMask-v0`

Adds binary foot-contact state (`feet_contact`) to the **actor** observation group. The baseline only exposes contact state to the Critic. With this mask the actor can condition actions on which feet are grounded, reducing foot slip.

```
PolicyCfg ← feet_contact = ObsTerm(func=mdp.current_feet_contact, ...)
```

**+ Adaptive Tracking** — `WalkRun-E1-AdaptiveTracking-v0`

Replaces the fixed-`std` linear velocity tracking reward with a contact-phase adaptive version. The `std` scales with the number of feet currently in contact:

| Contact state | std |
|---------------|-----|
| Double stance (2 feet) | `base_std` (strict) |
| Single stance (1 foot) | `base_std × 1.5` |
| Flight (0 feet) | `base_std × 2.0` (lenient) |

```
RewardsCfg ← track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_adaptive_exp, ...)
```

**+ Curiosity Reward** — `WalkRun-E1-CuriosityReward-v0`

Activates the built-in RND (Random Network Distillation) module. A predictor network is trained to match a fixed random target; the L2 prediction error becomes an intrinsic reward that encourages exploration of novel states. The intrinsic reward weight decays linearly from 0.5 to 0.05 over the first 5 000 iterations.

Two changes are required simultaneously:
- `walkrun_cfg_curiosity_reward.py`: uncomments `rnd_state: RndObsCfg` to provide RND input observations.
- `agents/rsl_rl_amp_ppo_cfg_curiosity.py`: sets `rnd_cfg = RslRlRndCfg(...)` to instantiate the RND networks and optimizer.

### Train Ablations

```bash
# PBHC baseline
python scripts/rsl_rl/train.py --task=WalkRun-E1-v0 --num_envs=4096 --headless

# + Contact Mask
python scripts/rsl_rl/train.py --task=WalkRun-E1-ContactMask-v0 --num_envs=4096 --headless

# + Adaptive Tracking
python scripts/rsl_rl/train.py --task=WalkRun-E1-AdaptiveTracking-v0 --num_envs=4096 --headless

# + Curiosity Reward
python scripts/rsl_rl/train.py --task=WalkRun-E1-CuriosityReward-v0 --num_envs=4096 --headless
```

### Play Ablations

```bash
python scripts/rsl_rl/play.py --task=WalkRun-E1-ContactMask-v0      --num_envs=1
python scripts/rsl_rl/play.py --task=WalkRun-E1-AdaptiveTracking-v0 --num_envs=1
python scripts/rsl_rl/play.py --task=WalkRun-E1-CuriosityReward-v0  --num_envs=1
```

### Monitor with Tensorboard

```bash
# baseline and contact-mask / adaptive-tracking share the same experiment_name
tensorboard --logdir=logs/rsl_rl/walkrunCAMP

# curiosity reward uses a separate log directory
tensorboard --logdir=logs/rsl_rl/walkrunCAMP_curiosity
```

---

## Usage

### Visualize motion

Visualize the motion by updating the simulation with data from e1/datasets/motion_visualization.

```bash
python scripts/rsl_rl/play_animation.py --task=WalkRun12dof-E1-v0
python scripts/rsl_rl/play_animation.py --task=Mimic-E1-v0
```


### Train

Train the policy using AMP expert data from e1/datasets/motion_amp_expert.

```bash
python scripts/rsl_rl/train.py --task=WalkRun12dof-E1-v0 --num_envs=4096 --headless
python scripts/rsl_rl/train.py --task=Mimic-E1-v0 --num_envs=4096 --headless
```

### Play

Run the trained policy.

```bash
python scripts/rsl_rl/play.py --task=WalkRun12dof-E1-v0 --num_envs=1
python scripts/rsl_rl/play.py --task=Mimic-E1-v0 --num_envs=1
```

### Sim2Sim(MuJoCo)

Evaluate the trained policy in MuJoCo to perform cross-simulation validation.

Exported_policy/ contains pretrained policies provided by the project. When using the play script, trained policy is exported automatically and saved to path like logs/rsl_rl/run/[timestamp]/exported/policy.pt.
```bash
python scripts/sim2sim_12dof.py --policy logs/rsl_rl/walkrun12dof/[timestamp]/exported/policy.onnx --duration 100 
python scripts/sim2sim_mimic.py --policy logs/rsl_rl/mimic/[timestamp]/exported/policy.onnx --duration 100 --motion_file source/NoetixE1/NoetixE1/assets/datasets/mimic/dance1.npz
```

### Tensorboard
```bash
tensorboard --logdir=logs/walkrun
```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/NoetixE1"
        "<path-to-rsl-rl>/rsl_rl",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_tasks",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_mimic",
        "<path-to-isaac-lab>/IsaacLab/source/extensions",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_assets",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_rl",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab",
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
