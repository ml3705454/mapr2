# Multi-Agent Probabilistic Recursive Reasoning (MAPR2)
Multi-Agent Probabilistic Recursive Reasoning is a multi-agent reinforcement learning framework. The algorithms are based on the paper [PROBABILISTIC RECURSIVE REASONING FOR
MULTI-AGENT REINFORCEMENT LEARNING](https://openreview.net/pdf?id=rkl6As0cF7) in ICLR 2019.


The learning path of PR2-AC in differential game:

![PR2AC](./figures/PR2AC_3D_90.gif)

## Local Installation

To get the environment installed correctly, you will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

1. Clone rllrb
  
 ```shell
cd <installation_path_of_your_choice>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
sudo pip3 install -e .
 ```

 2. Intsall other dependencies
   
 ```shell
sudo pip3 install joblib,path.py,gtimer,theano,keras,tensorflow,gym, tensorflow_probability
 ```

 3. Intsall maci
   
 ```shell
cd maci
sudo pip3 install -e .
 ```


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

