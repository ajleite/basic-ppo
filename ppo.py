#!/usr/bin/python3

# Copyright 2019 Abe Leite
# Based on "Proximal Policy Optimization Algorithms", Schulman et al 2017
# For the benefit of my fellow CSCI-B 659 students
# While I hope that this code is helpful I will not vouch for its total accuracy;
# my primary aim here is to elucidate the ideas from the paper.

import sys

import tensorflow as tf
import gym

ACTORS = 8
N_CYCLES = 10000
LEARNING_RATE = 0.00025
CYCLE_LENGTH = 128
BATCH_SIZE = CYCLE_LENGTH*ACTORS
CYCLE_EPOCHS = 3
MINIBATCH = 32*ACTORS
GAMMA = 0.99
EPSILON = 0.1

class DiscretePPO:
    def __init__(self, V, pi):
        ''' V and pi are both keras (Sequential)s.
            V maps state to single scalar value;
            pi maps state to discrete probability distribution on actions. '''
        self.V = V
        self.pi = pi
        self.old_pi = tf.keras.models.clone_model(self.pi)
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    @tf.function
    def pick_action(self, S):
        return tf.random.categorical(self.pi(tf.expand_dims(S,axis=0)), 1)[0,0]
    @tf.function
    def train_minibatch(self, SARTS_minibatch):
        S, A, R, T, S2 = SARTS_minibatch
        next_V = tf.where(T, tf.zeros((MINIBATCH,)), self.V(S2))
        next_V = tf.stop_gradient(next_V)
        advantage = R + GAMMA * next_V - self.V(S)
        V_loss = tf.reduce_sum(advantage ** 2)

        V_gradient = tf.gradients(V_loss, self.V.weights)
        self.optimizer.apply_gradients(zip(V_gradient, self.V.weights))

        ratio = tf.gather(self.pi(S), A, axis=1) / tf.gather(self.old_pi(S), A, axis=1)
        confident_ratio = tf.clip_by_value(ratio, 1-EPSILON, 1+EPSILON)
        current_objective = ratio * advantage
        confident_objective = confident_ratio * advantage
        PPO_objective = tf.where(current_objective < confident_objective, current_objective, confident_objective)
        PPO_objective = tf.reduce_mean(PPO_objective)

        pi_gradient = tf.gradients(-PPO_objective, self.pi.weights)
        self.optimizer.apply_gradients(zip(pi_gradient, self.pi.weights))
    @tf.function
    def train(self, SARTS_batch):
        S, A, R, T, S2 = SARTS_batch
        for _ in range(CYCLE_EPOCHS):
            # shuffle and split into minibatches!
            shuffled_indices = tf.random.shuffle(tf.range(BATCH_SIZE))
            num_mb = BATCH_SIZE // MINIBATCH
            for minibatch_indices in tf.split(shuffled_indices, num_mb):
                mb_SARTS = (tf.gather(S, minibatch_indices),
                    tf.gather(A, minibatch_indices),
                    tf.gather(R, minibatch_indices),
                    tf.gather(T, minibatch_indices),
                    tf.gather(S2, minibatch_indices))
                self.train_minibatch(mb_SARTS)

        for old_pi_w, pi_w in zip(self.old_pi.weights, self.pi.weights):
            old_pi_w.assign(pi_w)

def train_PPO(agent, envs, render=False):
    episode_returns = []
    current_episode_returns = [0 for env in envs]
    last_s = [env.reset() for env in envs]
    for _ in range(N_CYCLES):
        SARTS_samples = []
        next_last_s = []
        next_current_episode_returns = []
        for env, s, episode_return in zip(envs, last_s, current_episode_returns):
            for _ in range(CYCLE_LENGTH):
                a = agent.pick_action(s).numpy()
                s2, r, t, _ = env.step(a)
                if render:
                    env.render()
                episode_return += r
                SARTS_samples.append((s,a,r,t,s2))

                if t:
                    episode_returns.append(episode_return)
                    print(f'Episode {len(episode_returns):3d}: {episode_return}')
                    episode_return = 0
                    s = env.reset()
                else:
                    s = s2
            next_last_s.append(s)
            next_current_episode_returns.append(episode_return)
        last_s = next_last_s
        current_episode_returns = next_current_episode_returns

        SARTS_batch = [tf.stack(X, axis=0) for X in zip(*SARTS_samples)]
        agent.train(SARTS_batch)

def make_agent(env):
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    V = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=obs_shape),
                             tf.keras.layers.Dense(400, activation='relu'),
                             tf.keras.layers.Dense(300, activation='relu'),
                             tf.keras.layers.Dense(1)])
    pi = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=obs_shape),
                             tf.keras.layers.Dense(400, activation='relu'),
                             tf.keras.layers.Dense(300, activation='sigmoid'),
                             tf.keras.layers.Dense(n_actions, activation='softmax')])
    return DiscretePPO(V, pi)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ppo.py <Env-V*> (--render)')
    envs = [gym.make(sys.argv[1]) for _ in range(ACTORS)]
    agent = make_agent(envs[0])
    train_PPO(agent, envs, '--render' in sys.argv)
