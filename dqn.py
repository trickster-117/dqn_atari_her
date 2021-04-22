# Disclaimer:
# This code was adapted from the DQN example of Keras
# https://keras.io/examples/rl/deep_q_network_breakout/

# It uses a modified version of the baselines package found here:
# https://github.com/openai/baselines

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-s", "--seed", action="store", dest="seed", type=int, default=42)
parser.add_argument("-g", "--gpu", action="store", dest="gpu", type=int, default=1)
parser.add_argument("-n", "--name", action="store", dest="name", default="her")
args = parser.parse_args()
SEED = args.seed
GPU = args.gpu
NAME = args.name
import numpy as np

np.random.seed(SEED)
import tensorflow as tf
tf.compat.v1.set_random_seed(SEED)
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
from datetime import datetime
import time
import os
# Organizational info
INFO = NAME + str(SEED)
MODE = "home"
VIS = False
if MODE == "remote":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)  # Tesla 1 or Tesla 2
else:
    import vis_functions as vf

print("Info:", INFO)
print("Seed:", SEED)
print("Gpu:", GPU)
print("Mode:", MODE)

# GPU reserves only as much memory as it needs but doesnt release memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print(gpus[0].name)

# Hyperparameters
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
        epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

# Use the Baseline Atari environment because of Deepmind helper functions
# env = make_atari("BreakoutNoFrameskip-v4")
env = make_atari("MsPacmanNoFrameskip-v4")
# env = make_atari("MontezumaRevengeNoFrameskip-v0")
num_actions = env.action_space.n
# Wrap the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True, episode_life=True, clip_rewards=False)
env.seed(SEED)
test_every_episodes = 500

"""
## Train
"""
# In the Deepmind paper they use RMSProp however the Adam optimizer improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
frame_count = 0

# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

# LOG
date_and_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
log_folder = "logs/" + INFO + date_and_time
# Create tensorboard writer and set folder to store log in
pathlib.Path(log_folder).mkdir(exist_ok=True, parents=True)
writer = tf.summary.create_file_writer(logdir=log_folder)


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


def test_model(test_env, test_epsilon=0.05, test_episodes=30):
    sum_reward = 0
    for test_e in range(test_episodes):  # Run until solved
        test_state = np.array(test_env.reset())
        test_episode_reward = 0

        for t in range(1, max_steps_per_episode):
            # env.render();Adding this line would show the attempts
            # of the agent in a pop up window.

            if MODE == "home":
                test_env.render()

            # Use epsilon-greedy for exploration
            if test_epsilon > np.random.rand(1)[0]:
                # Take random action
                a = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                test_state_tensor = tf.convert_to_tensor(test_state)
                test_state_tensor = tf.expand_dims(test_state_tensor, 0)
                a_probs = model(test_state_tensor, training=False)
                # Take best action
                a = tf.argmax(a_probs[0]).numpy()

            # Apply the sampled action in our environment
            _, test_state_next, test_reward, is_done, _ = test_env.step(a)
            test_state_next = np.array(test_state_next)

            test_episode_reward += test_reward
            test_state = test_state_next

            if is_done:
                sum_reward += test_episode_reward
                out = str("Test episode: " + str(test_e) + " reward: " + str(test_episode_reward))
                print(out)
                with open(log_folder + "/log.txt", 'a') as f:
                    f.write(out + "\n")
                break

    mean_reward = sum_reward / test_episodes
    out = str("Mean test reward: " + str(mean_reward) + " reward: " + str(mean_reward))
    print(out)
    with open(log_folder + "/log.txt", 'a') as f:
        f.write(out + "\n")

    tf.summary.scalar(name="Mean test reward", data=mean_reward, step=e)
    writer.flush()


# The first model makes the predictions for Q-values which are used to
model = create_q_model()
model_target = create_q_model()
# Init with the same weights
model_target.set_weights(model.get_weights())

# Training loop
with writer.as_default():
    tf.summary.text(name="seed", data=str(SEED), step=0)
    writer.flush()
    with open(log_folder + "/log.txt", 'a') as f:
        f.write("Seed = " + str(SEED) + "\n")

    e = 0
    while True:  # Go through episodes
        frame_start_e = frame_count
        t_e_start = time.time()
        state = np.array(env.reset())
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            if MODE == "home":
                env.render()  # Adding this line would show the attempts of the agent in a pop up window.

            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
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
            if frame_count > epsilon_random_frames:
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            _, state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count > epsilon_random_frames and frame_count % update_after_actions == 0 and len(
                    done_history) > batch_size:

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                max_next_Q = tf.reduce_max(future_rewards, axis=1)
                updated_q_values = rewards_sample + gamma * max_next_Q

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

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            # 11) Log
            if frame_count % 1000 == 0 and frame_count > epsilon_random_frames:
                dqn_variable = model.trainable_variables
                tf.summary.histogram(name="dqn_variables", data=tf.convert_to_tensor(dqn_variable[0]),
                                     step=frame_count)
                tf.summary.histogram(name="gradients", data=tf.convert_to_tensor(grads[0]),
                                     step=frame_count)
                target_variable = model_target.trainable_variables
                tf.summary.histogram(name="target_variables", data=tf.convert_to_tensor(target_variable[0]),
                                     step=frame_count)
                writer.flush()

            if frame_count % 1000 == 0 and frame_count > epsilon_random_frames:
                # Visualize scalars of errors
                tf.summary.scalar(name="Mean bellman error", data=loss, step=frame_count)
                tf.summary.scalar(name="Bellman target (mean next state max Q)",
                                  data=mean_next_max_Q, step=frame_count)
                writer.flush()

            if done:
                t_e_end = time.time()
                t_elapsed = t_e_end - t_e_start
                frame_end_e = frame_count
                frames_proc = frame_end_e - frame_start_e
                time_per_frame = t_elapsed/frames_proc

                output = str(
                    "e: " + str(e) + " epsilon: " + str(epsilon) + " reward: " + str(episode_reward)
                    + " frame count: " + str(frame_count) + " time per frame: " + str(time_per_frame)
                    + " time_elapsed: " + str(t_elapsed) + "s")
                # Print and write to log
                print(output)
                with open(log_folder + "/log.txt", 'a') as f:
                    f.write(output + "\n")

                tf.summary.scalar(name="reward", data=episode_reward, step=e)
                tf.summary.scalar(name="epsilon", data=epsilon, step=e)
                tf.summary.scalar(name="time per frame", data=time_per_frame, step=e)
                tf.summary.text(name="log", data=output, step=e)
                writer.flush()

                # Iterate episode
                e += 1

                if e % test_every_episodes == 0:
                    pathlib.Path(log_folder + '/Models').mkdir(exist_ok=True)
                    model_filename = "dqn_" + str(e) + ".h5"
                    model.save(log_folder + '/Models/' + model_filename)

                    print("Testing")
                    test_model(env)
                break
