import numpy as np
import random                # Handling random number generation
import time                  # Handling time calculation
import cv2
import itertools as it

import torch
from vizdoom import *        # Doom Environment
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import namedtuple, deque
import math
from rl_plotter.logger import Logger, CustomLogger

import sys
sys.path.append('../../')
from algos.agents import DQNAgent
from algos.models import InnateValuesDUCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)



'''Create environment'''
def create_environment():
    game = DoomGame()
    game.set_episode_timeout(3000)

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map04")

    '''Load the correct configuration'''
    # original version
    # game.load_config("cgames/04_doom_corridor/doom_files/deadly_corridor.cfg")
    # game.load_config("cgames/03_doom_defend_center/doom_files/defend_the_center.cfg")

    # updated version
    # game.load_config("/home/rickyang/Documents/projects/IVRL/VIZDoom/scenarios/deadly_corridor.cfg")
    # game.load_config("/home/rickyang/Documents/projects/IVRL/VIZDoom/scenarios/defend_the_center.cfg")
    # game.load_config("/home/rickyang/Documents/projects/IVRL/VIZDoom/scenarios/defend_the_line.cfg")
    # game.load_config("/home/rickyang/Documents/projects/IVRL/VIZDoom/scenarios/basic.cfg")

    """Load the correct scenario"""
    # original version
    # game.set_doom_scenario_path("cgames/04_doom_corridor/doom_files/deadly_corridor.wad")
    # game.set_doom_scenario_path("cgames/03_doom_defend_center/doom_files/defend_the_center.wad")

    # updated version
    # game.set_doom_scenario_path("/home/rickyang/Documents/projects/IVRL/VIZDoom/scenarios/deadly_corridor.wad")
    # game.set_doom_scenario_path("/home/rickyang/Documents/projects/IVRL/VIZDoom/scenarios/defend_the_center.wad")
    # game.load_config("/home/rickyang/Documents/projects/IVRL/VIZDoom/scenarios/defend_the_line.cfg")
    # game.load_config("/home/rickyang/Documents/projects/IVRL/VIZDoom/scenarios/basic.cfg")

    # Set screen solution
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    # game.set_window_visible(False)
    # game.set_episode_timeout(1000)
    # game.set_episode_start_time(10)

    # # Set action set 1
    # game.set_available_buttons(
    #     [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_FORWARD,
    #      Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK]
    # )

    # Set action set 2
    game.set_available_buttons(
        [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_FORWARD,
         Button.MOVE_BACKWARD, Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK]
    )

    # # Set action set 3
    # game.set_available_buttons(
    #     [Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK]
    # )

    # # Set action set 4
    # game.set_available_buttons(
    #     [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.ATTACK]
    # )

    # # Set action set 5
    # game.set_available_buttons(
    #     [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK]
    # )

    # Here our possible actions
    # possible_actions = np.identity(7, dtype=int).tolist()
    # possible_actions = np.identity(6, dtype=int).tolist()
    # possible_actions.extend([[0, 0, 1, 0, 1, 0],
    #                          [0, 0, 1, 0, 0, 1],
    #                          [1, 0, 1, 0, 0, 0],
    #                          [0, 1, 1, 0, 0, 0]])

    n = game.get_available_buttons_size()
    possible_actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Set utilities
    # game.set_available_game_variables(
    #     [GameVariable.HEALTH, GameVariable.ARMOR, GameVariable.DAMAGECOUNT, GameVariable.KILLCOUNT]
    # )

    game.set_available_game_variables(
        [GameVariable.HEALTH, GameVariable.AMMO2, GameVariable.KILLCOUNT]
    )

    # game.set_living_reward(0.001)

    return game, possible_actions

'''Viewing our Enviroment'''
# print("The size of frame is: (", game.get_screen_height(), ", ", game.get_screen_width(), ")")
# print("No. of Actions: ", possible_actions)
# game.init()
# plt.figure()
# plt.imshow(game.get_state().screen_buffer.transpose(1, 2, 0))
# plt.title('Original Frame')
# plt.show()
# game.close()

game, possible_actions = create_environment()

'''Execute the code cell below to play Pong with a random policy.'''
# def random_play():
#     game.init()
#     game.new_episode()
#     score = 0
#     while True:
#         reward = game.make_action(possible_actions[np.random.randint(7)])
#         done = game.is_episode_finished()
#         score += reward
#         time.sleep(0.01)
#         if done:
#             print("Your total score is: ", score)
#             game.close()
#             break
# random_play()

'''Preprocessing Frame'''
# game.init()
# plt.figure()
# plt.imshow(preprocess_frame(game.get_state().screen_buffer.transpose(1, 2, 0),(0, -60, -40, 60), 84), cmap="gray")
# game.close()
# plt.title('Pre Processed image')
# plt.show()

'''Stacking Frame'''
def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (0, -60, -40, 60), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

'''Creating our Agent'''
INPUT_SHAPE = (4, 84, 84)
NEEDS_SIZE = game.get_available_game_variables_size() + 1
ACTION_SIZE = len(possible_actions)
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 32        # Update batch size
LR = 0.0001            # learning rate
TAU = .1               # for soft update of target parameters
UPDATE_EVERY = 100     # how often to update the network
UPDATE_TARGET = 10000  # After which thershold replay to be started
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 100        # Rate by which epsilon to be decayed

agent = DQNAgent(INPUT_SHAPE, NEEDS_SIZE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, InnateValuesDUCnn)

'''Watching untrained agent play'''
# game.init()
# score = 0
# state = stack_frames(None, game.get_state().screen_buffer.transpose(1, 2, 0), True)
# while True:
#     action, _, _ = agent.act(state)
#     score += game.make_action(possible_actions[action])
#     done = game.is_episode_finished()
#     if done:
#         print("Your total score is: ", score)
#         break
#     else:
#         state = stack_frames(state, game.get_state().screen_buffer.transpose(1, 2, 0), False)
#
# game.close()

'''Loading Agent'''
start_epoch = 0
scores = []
scores_window = deque(maxlen=100)

'''Train the Agent with Innate Values DQN'''
epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

# plt.plot([epsilon_by_epsiode(i) for i in range(1000)])

# Initialize rl_plotter parameters
logger = Logger(log_dir='/Data/VIZDoom_IVRL_DQN/arena', exp_name='vizdoom', env_name='myenv', seed=0)
# custom_logger = logger.new_custom_logger(filename="output.csv", fieldnames=["Living_weight", "Health_weight", "Armor_weight",
#                                                                             "Damage_weight", "Kill_enemy_weight", "Episode_rewards"])
custom_logger = logger.new_custom_logger(filename="output.csv", fieldnames=["Environment_weight", "Health_weight", "Ammo_weight", "Kill_enemy_weight", "Episode_rewards", "Loss"])

def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    game.init()
    utility_units_amount = np.array([100, 200, 1])
    frame_repeat = 2

    for i_episode in range(start_epoch + 1, n_episodes + 1):
        game.new_episode()
        # state = stack_frames(None, game.get_state().screen_buffer.transpose(1, 2, 0), True)
        state = stack_frames(None, game.get_state().screen_buffer, True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        while True:
            start_state = game.get_state()
            start_utility = start_state.game_variables
            needs_weight, action = agent.act(state, eps)
            env_reward = game.make_action(possible_actions[action], frame_repeat)
            done = game.is_episode_finished()
            if done:
                break
            else:
                original_next_state = game.get_state()
                # next_state = stack_frames(state, original_next_state.screen_buffer.transpose(1, 2, 0), False)
                next_state = stack_frames(state, original_next_state.screen_buffer, False)
                next_utility = original_next_state.game_variables
                delta_utility = ((next_utility - start_utility) / utility_units_amount)
                delta_utility = np.insert(delta_utility, 0, [env_reward/400])
                # delta_utility[1] = abs(delta_utility[1])
                reward = agent.get_innate_values_rewards(delta_utility, needs_weight)
                agent.step(state, needs_weight, action, delta_utility, reward, next_state, done)
                state = next_state

            score += reward

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        clear_output(True)
        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window) * 100, eps),
              end="")

        # Record the output
        custom_logger.update(
            # [needs_weight[0], needs_weight[1], needs_weight[2], needs_weight[3], needs_weight[4], score],
            # total_steps=i_episode
            [needs_weight[0], needs_weight[1], needs_weight[2], needs_weight[3], np.mean(scores_window) * 100, agent.loss], total_steps=i_episode)


    game.close()
    return scores


scores = train(10000)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()