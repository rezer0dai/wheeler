# Wheeler - Reinforcement Learning experimental framework ( Policy Gradients )
* WIP ~ work in progress, implementing + testing ideas
* Notes on Policy Gradients methods, related to this project : https://medium.com/reinforcement-learning-w-policy-gradients
* I focus mainly on Policy Gradients for now, and main engines implemented are DDPG + PPO, via ActorCritic ( full list of features you can read bellow )
* Key feature of this framework are : 
	* ***multi-agent + multi-PG-methods cooperation***
		* f.e. 3 agents, every of them different RL method : DDPG, PPO, DQN ( later mention not implemented yet )
		* form of cooperation : 
			* sharing experiences ( locals + central priority replay buffers )
			* 1 target model ( DPPG ), others explorers ( PPO ) :
				* same sized function approximators models ( bit part of it at least )
				* every X-time stamp sync explorer ( PPO ) with target ( DDPG ) - copy target agent weights to explorers agents ( most of the model, except last layers )
				* this way, sample efficient approaches like PPO can take current main agent, and try to explore environment by copying his behaviour ( while preserving some part of its own specific ~ last layers )
				* therefore we get more related experiences to our replay buffer, which are related to current main agent ( DDPG ) via sample efficient method ( PPO ) ~ theory : fast + efficient exploration of environment + more robust and independent experiences	- illustration code 
		```python
		stop_q = Queue() # signalizing when to stop training
		memory = coss_buffer(CFG) # our shared buffer 
		
		worker = Process( # our efficient explorer
			target = agent_launch,
			args = (1, PPO_CFG, ppo_callback, memory..))
		worker.start() # we can have multiple explorers ( or even targets )
		# now main agent, run it in main process
		agent_launch(0, DDPG_CFG, ddpg_callback, memory..)
		print("learning finished")
		...
		def ppo_callback(_, agent, _): # every round we sync with main agent
			agent.bot.alpha_sync(["neslayer_2_"])
			return False # we dont evaluate here
		def dppg_callback(task, _, scores): # just eval
			return task.goal_met(np.mean(scores, 0))
		```
		
		* testing environment ( not working yet, but idea presented is my main focus now ) : https://github.com/rezer0dai/wheeler/blob/master/experimental/her_coop.ipynb
	
	* **multi-task** 
		* f.e. separate 3D navigation to 3+ different reward functions
		* every sub-task has different critic ( and reward function ) but common actor ( can be detached )
			* detached : SoftActorCritic fashion, have target + explorer; while target is common, explorer can be per sub-task specific
 
*	From design perspective, this are some nice things you can be interested in to look at :
	* **Task management** : https://github.com/rezer0dai/wheeler/blob/master/utils/taskmgr.py
		* allowing { unity / openaigym } to work with uniformly at scale, multiple agents sampling trough parallel environments
	* **DDPG + PPO** ( + later on DQN and value function as well ) uniform implementation : https://github.com/rezer0dai/wheeler/blob/master/utils/policy.py
		* allow uniformly exchange those algorithms via minimal changes to learning algorithm/framework
		* same thing i plan to use when adding Q functions to framework
	* **Replay + Cross buffers** implementation : https://github.com/rezer0dai/wheeler/blob/master/utils/replay.py
		* allowing easy HER, GAE, RNN implementation ( past experience when using those are by time unusable, therefore need recompute ~ this is what we can do easily here )
		* cross buffer allows easy sharing experience between multiple agents + ability to reanalyze experience
		* however : if using this, it is somehow perf overkill. But good for experimenting with, have space for perf improvements
	* **Encoders** implementation : https://github.com/rezer0dai/wheeler/blob/master/utils/encoders.py + https://github.com/rezer0dai/wheeler/blob/master/utils/ac.py
		* RNN / CNN / RBF / .. as common layer for Actor + Critic
		* however, need improvements as total_size / out_size logic is bit broken ( workable with, but need to deeper rethinkg )
		* also i need to work on this bit more, as to rethink how to efficient share history, how to whole design more nice and generic, now it is just as PoC version
		* need to work on RNN, seems in my new design those not working as expected ~> TODO : find solution
- **DISACLAIMER** : now framework is too slow ( especially with replay buffer and reanalyzing ), and most of features mentioned {RNN,multi-*,} altougth implemented but not properly tested; all benchmark implementations was tested on simple DPPG setup with NoisyNetwork and N-step/GAE and SoftActorCritic ( well i just refactor whole design, so make clean benchmarks was first milestone, next on is work on features; first to go : RNN, second Reacher benchmark implementation, third Reacher HER implementation, forth Reacher + HER + multi bot, fifth Reacher + HER + multi bot + multi task, and so on )

- RESEARCH vs overkill  : given number of features ( later part ), and it performance, it may look like more proper name for framework should be "overkiller". Main goal of this project is to test ideas and build some environment where i can interchangeably to turn them on or off while core should works. on the other side my benchmark implementation of Reacher ( 20 arms, no HER, just advanatage, n-step, DDPG with shared replay buffer ~ and so therefore fast ) you can find at : https://github.com/rezer0dai/ReacherBenchmarkDDPG


Benchmark Implementations 
===
* OpenAI gym benchmarks :
	* cartpole : https://github.com/rezer0dai/wheeler/blob/master/envs/softmax.ipynb
		* Problem :  Policy Gradients to discrete actions
		* Solution : SoftMax w/ DDPG
	*  mountaincar : https://github.com/rezer0dai/wheeler/blob/master/envs/reward_tuning.ipynb
		* Problem : low state representation + reward function
		* Solution : RBF encoding + updating reward function
	*  acrobot : https://github.com/rezer0dai/wheeler/blob/master/envs/her.ipynb
		* Problem : Robotic control + sparse reward function
		* Solution : HER
	*  pendulum : https://github.com/rezer0dai/wheeler/blob/master/envs/continuous.ipynb
		* Problem : classic continuous PG
		* Solution : DDPG / PPO ~ nice problem for testing if algo works
		
Fatures & Ideas of Wheeler experiment : 
===

0. Soft Actor-Critic, Off-Policy, Asynchronous, DDPG, PPO, vanilla PG, td-lambda, GAE, GRU/LSTM, Noisy Networks, HER, RBF, state stacking, OpenAi-Gym + Unity-MlAgents, PyTorch

1. **multiple agents** ( with potentially different algorithms DDPG + PPO ), working together ( envs/experimental/reacher_her.py )   
2. multiple critics for one actor, every critic ( task / simulation ) has its own environment to test actor in parallel
3. **MULTIPLE GOALS** : Via separating critics, you are able to define multiple reward functions ( preferable as sparse as possible )
	* imagine 3D navigation, therefore you can separate it to 3 reward function each rewards how far on particular dimension
	* or task where is most important end of episode rather the start, while start also important ( lunar lander )
	* reach + fetch + move tasks
4. Noisy Networks for exploration ( as head after RNN )
5. Replay buffer with support for HER by default ( and curiosity as priority weights + decider what to forgot ~ better say if new experience is worth of remembering )
6. Attention mechanism (*Experimental + not properly tested yet*) on reward gradients ( for multi-ple simulations with different rewards functions ) applied for learning actor
7. RNN as actor network - GRU and also LSTM ( can be done also for critic ) for enriching markovian state for RL
8. easy to adapt encoding on top of it : RBF sampler, pre-trained CNN, normalizers 
9. built in support for normalization of states ( see openai implementation of Reacher )

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
   * jupyter notebook + test envs/*.ipynb


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
