import argparse
import gym
import numpy as np
import pickle
import tensorflow as tf

MAX_STEPS = 100

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    features = data['observations']
    labels = data['actions']
    labels = np.squeeze(labels, axis=1) # remove extra middle dimension

    # ensure number of features and labels the same
    assert features.shape[0] == labels.shape[0]

    return features, labels

def build_dataset(features, labels, batch_size=32):
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    iterator = dataset.make_initializable_iterator()

    return iterator, features_placeholder, labels_placeholder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, default="Reacher-v2")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_units', type=int, default=128)
    args = parser.parse_args()

    environment = args.environment
    batch_size = args.batch_size
    num_units = args.num_units

    features, labels = load_data(f"./expert_data/{environment}.pkl")
    iterator, features_placeholder, labels_placeholder = build_dataset(features, labels, batch_size)

    x, y = iterator.get_next()

    hidden1 = tf.layers.dense(x, units=num_units, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, units=num_units, activation=tf.nn.relu)
    y_pred = tf.layers.dense(hidden2, units=labels.shape[1])
    
    loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train = optimizer.minimize(loss)

    num_batches = features.shape[0] // batch_size

    env = gym.make(environment)
    rewards = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})

        for i in range(args.num_epochs):
            for j in range(num_batches):
                _, loss_value = sess.run((train, loss))
                if j % 100 == 0: print(f"Step {(i * num_batches) + j} Loss: {loss_value}")

        for _ in range(20):
            obs = env.reset()
            step = 0
            total_reward = 0
            
            while True:
                action = sess.run(y_pred, feed_dict={x: np.expand_dims(obs, axis=0)})
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                step += 1

                env.render()

                if done or step >= MAX_STEPS:
                    rewards.append(total_reward)
                    break
                    
        print(f"Mean reward: {np.mean(rewards)} ({np.std(rewards)} standard deviation)")
            
if __name__ == '__main__':
    main()