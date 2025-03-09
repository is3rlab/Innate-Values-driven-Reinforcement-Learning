# Innate-Values-driven RL Based Cognitive Modeling

## Abstract
Innate values describe agents' intrinsic motivations, which reflect their inherent interests and preferences for pursuing goals and drive them to develop diverse skills that satisfy their various needs. Traditional reinforcement learning (RL) is learning from interaction based on the feedback rewards of the environment. However, in real scenarios, the rewards are generated by agents' innate value systems, which differ vastly from individuals based on their needs and requirements. In other words, considering the AI agent as a self-organizing system, developing its awareness through balancing internal and external utilities based on its needs in different tasks is a crucial problem for individuals learning to support others and integrate community with safety and harmony in the long term. To address this gap, we propose a new RL model termed innate-values-driven RL (IVRL) based on combined motivations' models and expected utility theory to mimic its complex behaviors in the evolution through decision-making and learning. Then, we introduce two IVRL-based models: IV-DQN and IV-A2C. By comparing them with benchmark algorithms such as DQN, DDQN, A2C, and PPO in the Role-Playing Game (RPG) reinforcement learning test platform VIZDoom, we demonstrated that the IVRL-based models can help the agent rationally organize various needs, achieve better performance effectively.

> Combined Motivations' Models:
    <div align = center>
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/1.png" height="120" alt="innate-values">
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/0.png" height="120" alt="innate-values">
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/gre.png" height="120" alt="innate-values">
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/6.png" height="120" alt="innate-values">
    </div>

## Approach Overview
We assume that all the AI agents (like robots) interact in the same working scenario, and their external environment includes all the other group members and mission setting. In contrast, the internal environment consists of individual perception components including various sensors (such as Lidar and camera), the critic module involving intrinsic motivation analysis and innate values generation, the RL brain making the decision based on the feedback of rewards and description of the current state (including internal and external) from the critic module, and actuators relating to all the manipulators and operators executing the RL brain's decisions as action sequence and strategies.

> The Proposed IVRL model based on Expected Utility Theory Models:
    <div align = center>
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/2.png" height="250" alt="innate-values">
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/3.png" height="200" alt="innate-values">
    </div>

> The architecture of the IVRL DQN and Actor-Critic models:
    <div align = center>
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/4.png" height="145" alt="innate-values">
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/5.png" height="145" alt="innate-values">
    </div>

## Evaluation through Simulation Studies
### Experiment Setting
Considering that the [VIZDoom](https://vizdoom.cs.put.edu.pl/) testbed can customize the experiment environment and define various utilities based on different tasks and cross-platform, we selected it to evaluate evaluate our IVRL model. We choose four scenarios: Defend the Center, Defend the Line, Deadly Corridor, and Arens, and compare our models with several benchmark algorithms, such as DQN, DDQN, A2C, and PPO. These models were trained on an NVIDIA GeForce RTX 3080Ti GPU with 16 GiB of RAM.

> Four Testing Scenarios (Defend the Center, Defend the Line, Deadly Corridor, and Arens):
    <div align = center>
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/defend_the_center.png" height="145" alt="innate-values">
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/defend_the_line.png" height="145" alt="innate-values">
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/deadly_corridor.png" height="145" alt="innate-values">
    <img src="https://github.com/is3rlab/Innate-Values-driven-Reinforcement-Learning/blob/main/figures/arena.png" height="145" alt="innate-values">
    </div>
