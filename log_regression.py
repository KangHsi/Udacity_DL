import tensorflow as tf
from ntmnist import dataset





pickle_file = 'notMNIST.pickle'
nmt_datasets=dataset(pickle_file)
reg_beta=0.0001

# sess = tf.InteractiveSession()
sess = tf.InteractiveSession()
# Create the model
x = tf.placeholder(tf.float32, [None,784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# y = tf.nn.softmax(tf.matmul(x, W))

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))+reg_beta*tf.nn.l2_loss(W)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
# train_step.run({x: input, y_: lab})
for i in range(10000):
    batch_xs, batch_ys = nmt_datasets.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: nmt_datasets.test_dataset, y_: nmt_datasets.test_labels}))