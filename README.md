# Multi-Agent Probabilistic Recursive Reasoning (MAPR2)
Multi-Agent Probabilistic Recursive Reasoning is a multi-agent reinforcement learning framework. The algorithms are based on the paper [PROBABILISTIC RECURSIVE REASONING FOR
MULTI-AGENT REINFORCEMENT LEARNING](https://openreview.net/pdf?id=rkl6As0cF7) submitted to the ICLR 2019.

## Local installation

To get the environment installed correctly, you will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

1. Clone rllab
```
cd <installation_path_of_your_choice>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

2. Install other dependencies via pip

## Examples

1. Matrix Game
```
./experiment/run_matrix_game.py
```

2. Differential Game
```
./experiment/run_different_agents.py
```
