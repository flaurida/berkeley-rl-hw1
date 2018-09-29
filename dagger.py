import argparse
import gym
import load_policy
import numpy as np
import tensorflow as tf
import tf_util

MAX_STEPS = 1000

def build_network(num_units, obs_shape, action_shape):
    x = tf.placeholder(tf.float32, shape=[None, obs_shape])
    y = tf.placeholder(tf.float32, shape=[None, action_shape])

    hidden1 = tf.layers.dense(x, units=num_units, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, units=num_units, activation=tf.nn.relu)
    y_pred = tf.layers.dense(hidden2, units=action_shape)

    loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(loss)

    return train_op, loss, x, y, y_pred

def generate_expert_data(policy_fn, environment, total_obs):
    obs_data = []
    action_data = []
    steps = 0

    env = gym.make(environment)
    obs = env.reset()

    with tf.Session():
        while len(obs_data) < total_obs:
            action = policy_fn(obs[None, :])
            obs, _, done, _ = env.step(action)

            obs_data.append(obs)
            action_data.append(action)
            steps += 1

            if done or steps >= MAX_STEPS:
                obs = env.reset()
                steps = 0

    return np.array(obs_data), np.squeeze(np.array(action_data), axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--environment', type=str, default="Reacher-v2")
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--num_units', type=int, default=128)
    parser.add_argument('--obsperstep', type=int, default=1000)
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    environment = args.environment

    print('Loading and building expert policy...')
    policy_fn = load_policy.load_policy(f"./experts/{environment}.pkl")
    print('Expert policy loaded!')

    all_obs_data, all_action_data = generate_expert_data(policy_fn, environment, args.obsperstep)
    reward_means = []
    reward_stds = []

    train, loss, x, y, y_pred = build_network(args.num_units, all_obs_data.shape[1], all_action_data.shape[1])
    env = gym.make(environment)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for rollout in range(args.num_rollouts):
            # Step 1: Train policy on expert-labeled data
            for epoch in range(num_epochs):
                num_batches = all_obs_data.shape[0] // batch_size
                ind_perm = np.random.permutation(all_obs_data.shape[0])
                ind_perm = ind_perm[:(num_batches * batch_size)].reshape((num_batches, batch_size))

                for batch in range(num_batches):
                    indices = ind_perm[batch]
                    x_batch, y_batch = all_obs_data[indices], all_action_data[indices]
                    _, loss_value = sess.run((train, loss), feed_dict={x: x_batch, y: y_batch})

                print(f"Loss after rollout {rollout + 1}, epoch {epoch + 1}: {loss_value}")
            
            # Step 2: Run policy to get dataset of observations
            obs_data = []
            rew_data = []
            steps = 0
            total_rew = 0
            obs = env.reset()

            while len(obs_data) < args.obsperstep:
                action = sess.run(y_pred, feed_dict={x: obs[None, :]})
                obs, rew, done, _ = env.step(action)

                obs_data.append(obs)
                total_rew += rew
                steps += 1

                if done or steps >= MAX_STEPS:
                    rew_data.append(total_rew)
                    obs = env.reset()
                    steps = 0
                    total_rew = 0
                    
            obs_data = np.array(obs_data)
            reward_means.append(np.mean(rew_data))
            reward_stds.append(np.std(rew_data))

            # Step 3: Ask expert to label generated dataset with actions
            action_data = policy_fn(obs_data)

            # Step 4: Aggregate old and new datasets
            all_obs_data = np.concatenate((all_obs_data, obs_data), axis=0)
            all_action_data = np.concatenate((all_action_data, action_data), axis=0)

        # Visualize learned policy    
        obs = env.reset()
        steps = 0

        for _ in range(100):
            action = sess.run(y_pred, feed_dict={x: obs[None, :]})
            obs, _, done, _ = env.step(action)
            steps += 1

            env.render()

            if done:
                obs = env.reset()
                steps = 0


    print("Mean rewards over time:", reward_means)
    print("Standard deviation of rewards:", reward_stds)


if __name__  == '__main__':
    main()