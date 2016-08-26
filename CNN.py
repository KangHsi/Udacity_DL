# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


#############visualization
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
# flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/logs', 'Summaries directory')

##########################################


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


image_size = 28
num_labels = 10
num_channels = 1 # grayscale



def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# def accuracy(predictions, labels):
#   return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
#           / predictions.shape[0])


batch_size = 64


# graph = tf.Graph()

# with graph.as_default():
def train():
    sess = tf.InteractiveSession()
    # Input data.
    with tf.name_scope('input'):
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(None, image_size, image_size, num_channels),name='x-input')
        image_shaped_input = tf.reshape(tf_train_dataset, [-1, 28, 28, 1])
        tf.image_summary('input', image_shaped_input, 10)
        keep_prob=tf.placeholder("float")
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels),name='y-input')
        # tf_valid_dataset = tf.constant(valid_dataset,name='valid_dataset')
        # tf_test_dataset = tf.constant(test_dataset,name='test_dataset')




#########################################################nomalized form of the layers in nn

    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(1.0, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def nn_conv_layer(input_tensor, patch_size, num_channels,output_depth, layer_name, biases=False,act=None, pool=None):
        """Reusable code for making a simple neural net layer.

    """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([patch_size,patch_size,num_channels,output_depth])
                # print ("weights:%s"%(weights.get_shape()))
                variable_summaries(weights, layer_name + '/weights')
            if (biases==True):
                with tf.name_scope('biases'):
                    biases = bias_variable([output_depth])
                    # print("biases:%s" % (biases.get_shape()))
                    variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('conv2d'):
                # print("input:%s" % (input_tensor.get_shape()))
                preactivate = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], padding='SAME')
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)
                print("preactivate:%s" % (preactivate.get_shape()))
            if (pool!=None):
                max_pool=pool(preactivate,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME',name='max_pool')
            if (act!=None):
                activations = act(max_pool+biases, 'activation')
                # tf.histogram_summary(layer_name + '/activations', activations)

            return preactivate
    ###############################################################


    # Variables.
    patch_size = 7
    depth = 32
    num_hidden = 64
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    # layer1_biases = tf.Variable(tf.zeros([depth]))
    # layer2_weights = tf.Variable(tf.truncated_normal(
    #     [patch_size, patch_size, depth, depth], stddev=0.1))
    # layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data):


        hidden1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        # max_out=tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                padding='SAME')
        # hidden1 = tf.nn.relu(max_out + layer1_biases)
        #output of layer1 is 28x28x32
        print ("hidden1:%s"%hidden1.get_shape())
        i_depth=int(hidden1.get_shape()[3])
        #inception
        hidden2_11=nn_conv_layer(hidden1,1,i_depth,24,'layer2_1')
        hidden2_3rd = nn_conv_layer(hidden1,1,i_depth, 12, 'layer2_3rd')
        hidden2_33=nn_conv_layer(hidden2_3rd,3,12,26,'layer2_3')
        hidden2_5rd=nn_conv_layer(hidden1,1,i_depth,8,'layer2_5rd')
        hidden2_55=nn_conv_layer(hidden2_5rd,5,8,12,'layer2_5')
        hidden2_pool=tf.nn.avg_pool(hidden1,[1,2,2,1],[1,1,1,1],padding='SAME',name='layer2_pool/p')
        hidden2_pool=nn_conv_layer(hidden2_pool,1,i_depth,12,'layer2_pool')
        # tensor t4 with shape [2, 3]
        #tf.shape(tf.concat(0, [t3, t4])) == > [4, 3]

        hidden2_out=tf.concat(3, [hidden2_11, hidden2_33,hidden2_55,hidden2_pool])
        print ("hidden2'shape:%s"%hidden2_out.get_shape())
        hidden2_max=tf.nn.max_pool(hidden2_out,[1,3,3,1],[1,2,2,1],padding='SAME',name='layer2_maxout')
        i_depth = int(hidden2_max.get_shape()[3])
        # inception
        hidden3_11 = nn_conv_layer(hidden2_max, 1, i_depth, 24, 'layer3_1')
        hidden3_3rd = nn_conv_layer(hidden2_max, 1, i_depth, 12, 'layer3_3rd')
        hidden3_33 = nn_conv_layer(hidden3_3rd, 3, 12, 26, 'layer3_3')
        hidden3_5rd = nn_conv_layer(hidden2_max, 1, i_depth, 8, 'layer3_5rd')
        hidden3_55 = nn_conv_layer(hidden3_5rd, 5, 8, 12, 'layer3_5')
        hidden3_pool = tf.nn.avg_pool(hidden2_max, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME', name='layer3_pool/p')
        hidden3_pool = nn_conv_layer(hidden3_pool, 1, i_depth, 12, 'layer3_pool')
        hidden3_out = tf.concat(3, [hidden3_11, hidden3_33, hidden3_55, hidden3_pool])
        print("hidden3'shape:%s" % hidden3_out)
        hidden3_max = tf.nn.max_pool(hidden3_out, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='layer3_maxout')

        hidden4=tf.nn.avg_pool(hidden3_max,[1,7,7,1],[1,1,1,1],padding='VALID',name='layer4_avgpool')
        hidden4=tf.nn.dropout(hidden4,keep_prob,name='layer4_dropout')
        out=flatten(hidden4)
        # hidden4 = tf.squeeze(hidden4)
        layer4_weights = tf.Variable(tf.truncated_normal(
            [74, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        out=tf.nn.xw_plus_b(out,layer4_weights,layer4_biases)


        return out

    def flatten(inputs, scope=None):
        """Flattens the input while maintaining the batch_size.
          Assumes that the first dimension represents the batch.
        Args:
          inputs: a tensor of size [batch_size, ...].
          scope: Optional scope for op_scope.
        Returns:
          a flattened tensor with shape [batch_size, k].
        Raises:
          ValueError: if inputs.shape is wrong.
        """
        if len(inputs.get_shape()) < 2:
            raise ValueError('Inputs must be have a least 2 dimensions')
        dims = inputs.get_shape()[1:]
        k = dims.num_elements()
        with tf.op_scope([inputs], scope, 'Flatten'):

            return tf.reshape(inputs, [-1, k])

    # Training computation.
    logits = model(tf_train_dataset)
    train_prediction = tf.nn.softmax(logits)
    with tf.name_scope('cross_entropy'):
        diff = tf_train_labels * tf.log(train_prediction)
        print ("diff%s"%diff)
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.scalar_summary('cross entropy', cross_entropy)

    # logits = model(tf_train_dataset)
    # loss = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(
            FLAGS.learning_rate).minimize(cross_entropy)


    with tf.name_scope('accuracy'):
        print(tf_train_labels.get_shape())
        size=train_prediction.get_shape()[0]
        correct_prediction =  tf.equal(tf.argmax(train_prediction, 1) , tf.argmax(tf_train_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(_accuracy)
    tf.scalar_summary('accuracy', accuracy)


    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
    tf.initialize_all_variables().run()
    print('Initialized')

    num_steps = 10001
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,keep_prob:0.5}
        summary,_, loss, predictions = sess.run(
            [merged,train_step, cross_entropy, accuracy], feed_dict=feed_dict)

        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, loss))
            print('Minibatch accuracy: %f' % predictions)
            print('test accuracy: %f' % sess.run(accuracy, feed_dict={tf_train_dataset: test_dataset[0:1000],tf_train_labels:test_labels[0:1000],keep_prob:1.0}))

            test_writer.add_summary(summary, step)
        else:
            train_writer.add_summary(summary, step)
        # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))



def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()


# python .local/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py --logdir=/tmp/logs/train/
