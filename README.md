# Multi-Agent Probabilistic Recursive Reasoning (MAPR2)
Multi-Agent Probabilistic Recursive Reasoning is a multi-agent reinforcement learning framework. The algorithms are based on the paper [PROBABILISTIC RECURSIVE REASONING FOR
MULTI-AGENT REINFORCEMENT LEARNING](https://openreview.net/pdf?id=rkl6As0cF7) submitted to the ICLR 2019.


The learning path of PR2-AC in differential game:

![PR2AC](./figures/PR2AC_3D_90.gif)

## Local Installation

To get the environment installed correctly, you will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

1.Install rllab

```
cd <installation_path_of_your_choice>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

2.Install Multi-Agent Particle Environment

```
cd <installation_path_of_your_choice>
git clone https://github.com/openai/multiagent-particle-envs.git
cd multiagent-particle-envs
pip install -e .
```

3.Install other dependencies via pip


## Implemented Algorithms

- [X] PR2-AC/Q
- [x] DDPG
- [x] DDPG with Opponent Modelling
- [x] DDPG with Symplectic Gradient Adjustment Optimization
- [x] MADDPG
- [x] MASQL
- [x] IGA
- [x] WoLF-IGA/PHC
- [x] LOLA-Exact


## Examples

* Matrix Game
```
python3 ./experiment/run_matrix_game.py
python3 ./experiment/run_wolf_game_no_dependency.py
```

* Differential Game
```
python3 ./experiment/run_different_agents.py
python3 ./experiment/run_ddpg_sga.py
```

* Particle Games
```
python3 ./experiment/run_particle_different_agents.py
```
