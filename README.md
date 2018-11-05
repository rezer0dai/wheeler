# Wheeler - Policy Gradient based experimental framework
* WIP ~ work in progress, not properly tested, except openai gym classic environments, TODO.. *

Why name "Wheeler" ? 
===
+ i choose learning by reinventing wheel ( aka reimplementing other ideas ), however ...

Features & Ideas of Wheeler experiment : 
===

0. Soft Actor-Critic, Off-Policy, Asynchronous, DDPG, PPO, vanilla PG, td-lambda, GAE, GRU/LSTM, Noisy Networks, HER, RBF, state stacking, OpenAi-Gym + Unity-MlAgents, PyTorch

1. **multiple agents** ( with potentially different algorithms DDPG + PPO ), working together ( envs/experimental/reacher_her.py )   
2. multiple critics for one actor, every critic ( task / simulation ) has its own environment to test actor in parallel
2. **MULTIPLE GOALS** : Via separating critics, you are able to define multiple reward functions ( preferable as sparse as possible )
  * imagine 3D navigation, therefore you can separate it to 3 reward function each rewards how far on particular dimension
  * or task where is most important end of episode rather the start, while start also important ( lunar lander )
  * reach + fetch + move tasks

3. Noisy Networks for exploration ( as head after RNN )
4. Replay buffer with support for HER by default ( and curiosity as priority weights + decider what to forgot ~ better say if new experience is worth of remembering )
5. Attention mechanism (*Experimental + not properly tested yet*) on reward gradients ( for multi-ple simulations with different rewards functions ) applied for learning actor
6. RNN as actor network - GRU and also LSTM ( can be done also for critic ) for enriching markovian state for RL
7. easy to adapt encoding on top of it : RBF sampler, pre-trained CNN, normalizers 
8. built in support for normalization of states ( see openai implementation of Reacher )

X. other experiments : 
  + default SoftMax Policy implementation for discrete actions
  + MonteCarlo search configuration ( how many times to replay with same seed )
  + immidiate learning for latest experience ( can be postponed, learn very x-time stamp, learn y-times )
  + postponed learning, at the end of the episode, from replay buffer only ( from config, you can select how much )
  + select steps from episode you consider important ( good in task.py, and good_reach in cfg.toml ) to save to replay buffer
  + originally supposed to be ML framework independent ( pytorch + keras as backend, but when i move to DDPG, i move towards pytorch, in future i may again reintroduce keras/tensorflow )

...framework works against classic four ( Acro, Pendelum, MountainCart, CartPole ) and it is its baseline

### Requirements : 
  * pytorch
  * openai gym 
  ```
  pip install gym
  cd wheeler
  ```
  * toml : 
  ```pip install toml```
  * test ( pendelum ~230 eps, mountaincar with 300 time stemps cap : ~11 episodes, cartpole ~100 episodes, acro ~50 episodes ): 
  ```
  cd wheeler
  python envs/pendelum.py
  ```
    

## TODO : 
Blog on what i have larned from OpenAi Gym set : Acro, Pendelum, MountainCar, CartPole, and also from Reacher :
1. Vanila policy gradients on cartpole (1 - action) log vs distributions
2. DPPG vs Vanila PG on continuous : MountainCar and Pendelum ( problem with approach 1 while multiple choices, PG distrbutions sig+mu, and in general with continous spaces )
3. RBF encoding for better feature capturing
4. whats wrong with manual engineering of rewards function -> aka networks likely does not overestimate at all...
5. HER on acrobot
6. minute of thinking about ReplayBuffer ( with curiosity, with her, .. )
7. my view why ReLU in NN layers is likely to be enough ( introducing non-linearity while preserving fair-enough gradients .. idea when thinking about softmax on last layer ~ logits vs output )
8. MDP vs history on states ( multiple frames vs/combine RNN-alike-Memory )
9. multiple tasks vs multiple critics and different reward functions
10. multiple agents cooperating ( with potentionally different algorithms )

important references ( more to add ) : 
===
  + https://github.com/openai/baselines
  + https://github.com/Kaixhin/NoisyNet-A3C
  + https://github.com/vitchyr/rlkit
  + https://github.com/higgsfield/RL-Adventure-2
  + https://github.com/dennybritz/reinforcement-learning
  + https://github.com/vy007vikas/PyTorch-ActorCriticRL
  + https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
  + https://github.com/takoika/PrioritizedExperienceReplay
  + ...
  + step by step i will add more resources i am/was using
  + also need to link papers ( GAE, DDPG, SoftActor-Critic, ... ), TODO
