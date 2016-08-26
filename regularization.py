import tensorflow as tf
from ntmnist import dataset
import numpy as np
pickle_file = 'notMNIST.pickle'
data=dataset(pickle_file)
########
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
############

train_subset = 1000
image_size = 28
num_labels = 10
hn_num=1024
batch_size = 128
reg_beta=0.025


data.train_dataset=data.train_dataset[0:train_subset,:]
data.train_labels=data.train_labels[0:train_subset]

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(None, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))


    with tf.name_scope('hidden') as scope_1:
        # Variables.
        weights_1 = tf.Variable(
            tf.truncated_normal([image_size * image_size, hn_num]),name='weights')
        biases_1 = tf.Variable(tf.zeros([hn_num]),name='biases')
        print (tf.shape(weights_1))
        hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)


    # dropout on hidden layer
    keep_prob = tf.placeholder("float")
    hidden_layer_drop = tf.nn.dropout(hidden1, keep_prob)


    with tf.name_scope('out') as scope_2:
        # Variables.
        weights = tf.Variable(
            tf.truncated_normal([hn_num,num_labels]), name='weights')
        biases = tf.Variable(tf.zeros([num_labels]), name='biases')
        print (tf.shape(weights))
        logits = tf.matmul(hidden_layer_drop, weights) + biases


    # regularizer=[weights_1,weights]
    ##########################
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss = loss+reg_beta*(tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    # valid_prediction = tf.nn.softmax(
    #     tf.matmul(tf_valid_dataset, weights) + biases)
    # test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 5001
# num_steps=30

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (data.train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = data.train_dataset[offset:(offset + batch_size), :]
    batch_labels = data.train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
         session.run(train_prediction,feed_dict={tf_train_dataset: data.valid_dataset,keep_prob:1.0}), data.valid_labels))
      print(tf.nn.l2_loss(weights).eval())
  feed_dict = {tf_train_dataset: data.test_dataset,keep_prob:1.0}
  predictions=session.run(train_prediction, feed_dict=feed_dict)
  print("Test accuracy: %.1f%%" % accuracy(predictions, data.test_labels))

  # print(session.run(accuracy, feed_dict={tf_train_dataset: tf_test_dataset, tf_train_labels: data.test_labels}))