import numpy as np
import pickle
import tensorflow as tf

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
    features, labels = load_data('./expert_data/Reacher-v2.pkl')
    iterator, features_placeholder, labels_placeholder = build_dataset(features, labels)

    x, y_true = iterator.get_next()

    hidden1 = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    y_pred = tf.layers.dense(hidden1, units=2)
    
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})

        for i in range(500):
            _, loss_value = sess.run((train, loss))
            print(loss_value)
            
if __name__ == '__main__':
    main()