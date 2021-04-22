# Disclaimer:
# This code was adapted from the DQN example of Keras
# https://keras.io/examples/rl/deep_q_network_breakout/

# It uses a modified version of the baselines package found here:
# https://github.com/openai/baselines

# It was modified to use the Hindsight Experience Replay (HER) algorithm to cope with sparse rewards.
# The algorithm was first published in : https://arxiv.org/abs/1707.01495

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-s", "--seed", action="store", dest="seed", type=int, default=42)
parser.add_argument("-g", "--gpu", action="store", dest="gpu", type=int, default=1)
parser.add_argument("-n", "--name", action="store", dest="name", default="her")
parser.add_argument("-m", "--her", action="store", dest="her", default="final")
args = parser.parse_args()
SEED = args.seed
GPU = args.gpu
NAME = args.name
HER_MODE = args.her

import numpy as np

np.random.seed(SEED)
import tensorflow as tf
import random

random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime
import time
import os
import copy

# Organizational info
INFO = NAME + str(SEED)
MODE = "home"
VIS = False
if MODE == "remote":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)  # Tesla 1 or Tesla 2
else:
    import vis_functions as vf

# LOG
date_and_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
log_folder = "logs/" + INFO + date_and_time

# GPU reserves only as much memory as it needs but doesnt release memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Hyperparameters
K_HER = 4
STALL_LIMIT = 300
STALL_VAR = 3
STALL_CHECK_FREQ = 10
GAMMA = 0.99  # Discount factor for past rewards
EPSILON = 1.0  # Epsilon greedy parameter
EPSILON_MIN = 0.1  # Minimum epsilon greedy parameter
EPSILON_MAX = 1.0  # Maximum epsilon greedy parameter
EPSILON_INTERVAL = (EPSILON_MAX - EPSILON_MIN)  # Rate at which to reduce chance of random action being taken
BATCH_SIZE = 32  # Size of batch taken from replay buffer
MAX_STEPS_EPISODE = 5000  # Max frames per run, to prevent stalls if stall detection fails!
TEST_RATE = 500
EPS_RANDOM_FRAMES = 50000  # Number of frames to take random action and observe output
EPS_GREEDY_FRAMES = 1000000.0  # Number of frames for exploration
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
MAX_MEMORY_LENGTH = 100000
UPDATE_AFTER_ACTIONS = 4  # Train the model after 4 actions
UPDATE_TARGET_NET = 10000  # How often to update the target network

# Atari env
# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("MontezumaRevengeNoFrameskip-v0")
# env = make_atari("MsPacmanNoFrameskip-v4")
num_actions = env.action_space.n
# Wrap the frames, grey scale, stack four frames and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True, episode_life=True, clip_rewards=False)
env.seed(SEED)

# In the Deepmind paper they use RMSProp however then Adam optimizer improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []

frame_count = 0
frame_count_train = 0

# Episode buffer for HER = S
episode_state_buffer = []
episode_state_next_buffer = []
episode_action_buffer = []
episode_pos_buffer = []
episode_done_buffer = []

# Using huber loss for stability
loss_function = keras.losses.Huber()

# Create tensorboard writer and set folder to store log in
pathlib.Path(log_folder).mkdir(exist_ok=True, parents=True)
writer = tf.summary.create_file_writer(logdir=log_folder)
writer.flush()


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action_layer = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action_layer)


def test_model(test_epsilon=0.05, test_episodes=30):
    print("Testing:")
    reward_list = []
    t_all_tests = 0

    # if env.spec.id == "MontezumaRevengeNoFrameskip-v0":
    test_episode_pos_buffer = []

    for test_e in range(test_episodes):  # Run until solved
        test_state = np.array(env.reset())
        test_episode_reward = 0
        t_test_e_start = time.time()

        for t_test in range(1, MAX_STEPS_EPISODE):
            # env.render();Adding this line would show the attempts
            # of the agent in a pop up window.

            if MODE == "home":
                env.render()

            # Use epsilon-greedy for exploration
            if test_epsilon > np.random.rand(1)[0]:
                # Take random action
                a = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                test_state_tensor = tf.convert_to_tensor(test_state)
                test_state_tensor = tf.expand_dims(test_state_tensor, 0)
                q_values = model(test_state_tensor, training=False)
                # Take best action
                a = tf.argmax(q_values[0]).numpy()

            # Apply the sampled action in our environment
            frame_c, test_state_next, test_reward, test_done, _ = env.step(a)
            test_state_next = np.array(test_state_next)

            if env.spec.id == "MontezumaRevengeNoFrameskip-v0":
                test_pos = get_jones(frame_c)
                test_episode_pos_buffer.append(test_pos)

            if env.spec.id == "MontezumaRevengeNoFrameskip-v0":
                test_is_stalled = stall_detection(test_episode_pos_buffer)

                if test_is_stalled:
                    test_done = True
                    out = "Episode finished because of detected stall in test!"
                    do_log(out, log_folder)
                    env.env.env.env.env.was_real_done = True
                    env.reset()

            test_episode_reward += test_reward
            test_state = test_state_next

            if t_test == MAX_STEPS_EPISODE - 1:
                test_done = 1
                out = "Episode finished because of max frame limit per run!"
                do_log(out, log_folder)
                env.env.env.env.env.was_real_done = True
                env.reset()

            if test_done:
                t_test_e_end = time.time()
                t_ela = t_test_e_end - t_test_e_start
                t_all_tests += t_ela
                reward_list.append(test_episode_reward)
                out = str("Test episode: " + str(test_e) + " reward: " + str(test_episode_reward) + " time elapsed: "
                          + str(t_ela))
                do_log(out, log_folder)

                del test_episode_pos_buffer[:]

                break

    mean_reward = np.mean(reward_list)
    out = str("Mean test reward: " + str(mean_reward) + " reward: " + str(mean_reward) + " time for all tests: "
              + str(t_all_tests))
    do_log(out, log_folder)

    tf.summary.scalar(name="Mean test reward", data=mean_reward, step=e)
    tf.summary.scalar(name="Time per collection of tests", data=t_all_tests, step=e)
    writer.flush()


def stall_detection(pos_buffer):
    if len(pos_buffer) % STALL_CHECK_FREQ == 0 and len(pos_buffer) > STALL_LIMIT:
        stalled = False
        # If positions of x or y were the same for 100 frames
        x_pos = np.array(pos_buffer)[:, 0][-STALL_LIMIT:]
        y_pos = np.array(pos_buffer)[:, 1][-STALL_LIMIT:]

        x_pos_mean = np.mean(x_pos)
        y_pos_mean = np.mean(y_pos)

        if all(x_pos_mean-STALL_VAR < x < x_pos_mean+STALL_VAR for x in x_pos) or \
                all(y_pos_mean-STALL_VAR < y < y_pos_mean+STALL_VAR for y in y_pos):
            print("Stalled")
            stalled = True

        return stalled
    else:
        return False


def train_dqn(fr_count):
    # Get indices of samples for replay buffers
    indices = np.random.choice(range(len(done_history)), size=BATCH_SIZE)

    # Using list comprehension to sample from replay buffer
    state_sample = np.array([state_history[j] for j in indices])
    state_next_sample = np.array([state_next_history[j] for j in indices])
    rewards_sample = [rewards_history[j] for j in indices]
    action_sample = [action_history[j] for j in indices]
    done_sample = tf.convert_to_tensor(
        [float(done_history[j]) for j in indices]
    )

    # Show batch if reward is > 0
    if VIS:
        if rewards_sample[0] > 0 or rewards_sample[1] > 0 or rewards_sample[2] > 0 or rewards_sample[3] > 0:
            vf.vis_minibatch(state_sample, state_next_sample, action_sample, rewards_sample, done_sample)

    # Build the updated Q-values for the sampled future states, Use the target model for stability
    future_rewards = model_target.predict(state_next_sample)
    # Q value = reward + discount factor * expected future reward
    max_next_Q = tf.reduce_max(future_rewards, axis=1)
    updated_q_values = rewards_sample + GAMMA * max_next_Q

    mean_next_max_Q = tf.reduce_mean(max_next_Q)

    # If final frame set the last value to -1
    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

    # Create a mask so we only calculate loss on the updated Q-values
    masks = tf.one_hot(action_sample, num_actions)

    with tf.GradientTape() as tape:
        # Train the model on the states and updated Q-values
        q_values = model(state_sample)

        # Apply the masks to the Q-values to get the Q-value for action taken
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        # Calculate loss between new Q-value and old Q-value
        loss = loss_function(updated_q_values, q_action)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if fr_count % UPDATE_TARGET_NET == 0:
        # update the the target network with new weights
        model_target.set_weights(model.get_weights())

    # 11) Log
    if fr_count % 1000 == 0 and fr_count > EPS_RANDOM_FRAMES:
        dqn_variable = model.trainable_variables
        tf.summary.histogram(name="dqn_variables", data=tf.convert_to_tensor(dqn_variable[0]),
                             step=fr_count)
        tf.summary.histogram(name="gradients", data=tf.convert_to_tensor(grads[0]),
                             step=fr_count)
        target_variable = model_target.trainable_variables
        tf.summary.histogram(name="target_variables", data=tf.convert_to_tensor(target_variable[0]),
                             step=fr_count)
        tf.summary.histogram(name="reward_history", data=tf.convert_to_tensor(rewards_history), step=fr_count)
        writer.flush()

    if fr_count % 1000 == 0 and fr_count > EPS_RANDOM_FRAMES:
        # Visualize scalars of errors
        tf.summary.scalar(name="Mean bellman error", data=loss, step=fr_count)
        tf.summary.scalar(name="Bellman target (mean next state max Q)",
                          data=mean_next_max_Q, step=fr_count)
        writer.flush()


###############################################################################
# Extension ###################################################################
###############################################################################
def get_jones(frame):
    # plt.figure(1)
    # plt.imshow(frame)
    # plt.pause(0.001)

    jones_val = 72
    jones = np.zeros_like(frame[:, :, 0])

    # Get Jones
    # jones[frame[:, :, 1] != jones_val] = 0
    jones[frame[:, :, 1] == jones_val] = 1

    # cut out lives
    jones[0:20, :] = 0

    # Get mean coordinate
    y = np.mean(np.nonzero(jones)[0])
    x = np.mean(np.nonzero(jones)[1])

    # plt.figure(2)
    # plt.clf()
    # plt.imshow(jones, cmap="gray")
    # plt.scatter(x, y, c='red')
    # plt.pause(0.000001)
    # plt.show()

    return tuple([x, y])


def get_pacman(frame):

    # Channel values of pacman color = (210,164,74)
    pacman_val = 210
    pacman = np.zeros_like(frame[:, :, 0])

    # Get pacman
    pacman[frame[:, :, 0] != pacman_val] = 0
    pacman[frame[:, :, 0] == pacman_val] = 1

    # Get mean coordinate
    y = np.mean(np.nonzero(pacman)[0])
    x = np.mean(np.nonzero(pacman)[1])

    # plt.figure(1)
    # plt.clf()
    # plt.imshow(frame)
    # plt.scatter(x, y)
    # plt.pause(0.000001)

    return tuple([x, y])


def sample_goal(mode, k=4):
    goal = None
    # 1) For HER we store each transition in the replay buffer twice: once with the goal used for the generation
    # of the episode and once with the goal corresponding to the final state from the episode
    if mode == "final":
        # Take the last state in episode buffer
        # is still empty
        if len(episode_pos_buffer) == 0:
            goal = None
        else:
            goal = [episode_pos_buffer[-1]]
    elif mode == "future":
        goal = random.sample(episode_pos_buffer, k)

    return goal


def goal_reached(agent_pos, goal_pos, eps=0.001):
    # If position is approximately at the goal
    if np.linalg.norm(np.subtract(agent_pos, goal_pos)) < eps:
        return 10
    else:
        return 0


def do_log(out, folder):
    print(out)
    with open(folder + "/log.txt", 'a') as f:
        f.write(out + "\n")


do_log(str("Info: " + INFO), log_folder)
do_log(str("Seed: " + str(SEED)), log_folder)
do_log(str("GPU: " + str(GPU)), log_folder)
do_log(str("Mode: " + MODE), log_folder)
do_log(str("HER mode: " + HER_MODE), log_folder)

# START
# The first model makes the predictions for Q-values
model = create_q_model()
model_target = create_q_model()
# Init with the same weights
model_target.set_weights(model.get_weights())

# Main loop
with writer.as_default():
    # START LOGS
    tf.summary.text(name="seed", data=str(SEED), step=0)
    init_dqn_variable = model.trainable_variables
    tf.summary.histogram(name="dqn_variables", data=tf.convert_to_tensor(init_dqn_variable[0]),
                         step=0)
    init_target_variable = model_target.trainable_variables
    tf.summary.histogram(name="target_variables", data=tf.convert_to_tensor(init_target_variable[0]),
                         step=0)

    e = 0
    # Go through episodes
    while True:
        frame_start_e = frame_count
        t_e_start = time.time()

        # Init proper frame count for train phase for better log and control
        frame_count_train = copy.copy(frame_count)

        # Initialize state, # Original goal is given by env
        state = np.array(env.reset())

        episode_reward = 0
        is_stalled = False  # only for Montezumas revenge

        # 1) Normal loop for experience replay
        for timestep in range(1, MAX_STEPS_EPISODE):

            if MODE == "home":
                env.render()  # Adding this line would show the attempts of the agent in a pop up window.

            # Use epsilon-greedy for exploration
            if frame_count < EPS_RANDOM_FRAMES or EPSILON > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            if frame_count > EPS_RANDOM_FRAMES:
                EPSILON -= EPSILON_INTERVAL / EPS_GREEDY_FRAMES
                EPSILON = max(EPSILON, EPSILON_MIN)

            # Apply the sampled action in our environment
            frame_color, state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)
            episode_reward += reward

            if env.spec.id == "MontezumaRevengeNoFrameskip-v0":
                t1 = time.time()
                pos = get_jones(frame_color)
                t2 = time.time()
                print("Time needed:", t2-t1)
            elif env.spec.id == "MsPacmanNoFrameskip-v4":
                pos = get_pacman(frame_color)

            # Fill episode buffer S with frame and store positions
            episode_action_buffer.append(action)
            episode_state_buffer.append(state)
            episode_state_next_buffer.append(state_next)
            episode_done_buffer.append(done)
            # Reward will be calculated later
            # Position for HER
            episode_pos_buffer.append(pos)

            if env.spec.id == "MontezumaRevengeNoFrameskip-v0":
                is_stalled = stall_detection(episode_pos_buffer)

            # In all cases we also replay each trajectory with
            # the original goal pursued in the episode
            # = Standard experience replay
            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)

            # Limit the state and reward history
            if len(rewards_history) > MAX_MEMORY_LENGTH:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            # Set state = next
            state = state_next
            frame_count += 1

            if env.spec.id == "MontezumaRevengeNoFrameskip-v0":
                if is_stalled:
                    done = True
                    text = "Episode finished because of detected stall!"
                    do_log(text, log_folder)
                    env.env.env.env.env.was_real_done = True
                    env.reset()

            if timestep == MAX_STEPS_EPISODE - 1:
                done = True
                text = "Episode finished because of max frame limit per run!"
                do_log(text, log_folder)
                env.env.env.env.env.was_real_done = True
                env.reset()

            if done:
                tf.summary.scalar(name="reward", data=episode_reward, step=e)
                tf.summary.scalar(name="epsilon", data=EPSILON, step=e)
                writer.flush()

                if e % TEST_RATE == 0:
                    pathlib.Path(log_folder + '/Models').mkdir(exist_ok=True)
                    model_filename = "dqn_" + str(e) + ".h5"
                    model.save(log_folder + '/Models/' + model_filename)

                # Iterate episode
                e += 1
                break

        if env.spec.id == "MontezumaRevengeNoFrameskip-v0":
            # Delete elements where agent was stall. Dont learn with HER from them
            if is_stalled:
                episode_pos_buffer[-STALL_LIMIT:] = []

        # 2) Second loop for insertion of experience replay data by HER
        if HER_MODE == "final":
            goals = sample_goal(mode=HER_MODE)

        final_goal_reached = False
        for i, pos in enumerate(episode_pos_buffer):

            if HER_MODE == "future":
                goals = sample_goal(mode=HER_MODE, k=K_HER)

            # print("\nNew Step")
            # Add experience to replay memory for extra goals that were reached
            for g in goals:
                r_her = goal_reached(pos, g)

                # Save in replay buffers
                action_history.append(episode_action_buffer[i])
                state_history.append(episode_state_buffer[i])
                state_next_history.append(episode_state_next_buffer[i])
                done_history.append(episode_done_buffer[i])
                rewards_history.append(r_her)

                # Limit the state and reward history aswell with memory injected by HER
                if len(rewards_history) > MAX_MEMORY_LENGTH:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                # # TEST
                # if r_her != 0:
                #     print("Goal reached")
                # else:
                #     print("Goal not reached")

                if r_her != 0 and VIS:
                    vf.vis_one_memory_element(episode_state_buffer[i], episode_state_next_buffer[i],
                                              episode_action_buffer[i], r_her, 0)

                    plt.figure(2)
                    plt.clf()
                    conv_x = 84 / 160
                    conv_y = 84 / 210
                    plt.imshow(episode_state_next_buffer[i][:, :, 3], cmap="gray")
                    plt.scatter(pos[0] * conv_x, pos[1] * conv_y, c='r', )

                    for g_vis in goals:
                        plt.scatter(g_vis[0] * conv_x, g_vis[1] * conv_y, marker='*')

                    plt.pause(0.00001)
                    print("Goal reached")

                if r_her != 0 and HER_MODE == "final":
                    final_goal_reached = True

            if final_goal_reached:
                break  # otherwise a lot if rewards are being stored for the same position

        # 3) Training loop
        for t in range(len(episode_state_buffer)):

            frame_count_train += 1

            # Update every fourth frame and once batch size is over 32, limit depends on played frames,
            if frame_count > EPS_RANDOM_FRAMES and frame_count_train % UPDATE_AFTER_ACTIONS == 0:
                # Do Training
                train_dqn(frame_count_train)

        # Clear episode buffer and position buffer to be filled with new frames
        del episode_state_buffer[:]
        del episode_state_next_buffer[:]
        del episode_action_buffer[:]
        del episode_pos_buffer[:]
        del episode_done_buffer[:]

        # Final log
        t_e_end = time.time()
        t_elapsed = t_e_end - t_e_start
        frame_end_e = frame_count
        frames_proc = frame_end_e - frame_start_e
        time_per_frame = t_elapsed / frames_proc

        output = str(
            "e: " + str(e) + " epsilon: " + str(EPSILON) + " reward: " + str(episode_reward)
            + " frame count: " + str(frame_count) + " delta_frames: " + str(frames_proc)
            + " time_elapsed: " + str(t_elapsed) + "s")
        # Print and write to logs
        do_log(output, log_folder)

        tf.summary.scalar(name="delta frames", data=frames_proc, step=e - 1)
        tf.summary.scalar(name="time per frame", data=time_per_frame, step=e - 1)
        tf.summary.text(name="log", data=output, step=e - 1)
        writer.flush()

        # Testing
        if e % TEST_RATE == 0:
            test_model()
