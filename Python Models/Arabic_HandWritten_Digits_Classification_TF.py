import os
import os.path as path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
os.environ["CUDA_VISIBLE_DEVICES"]="0"    #for training on gpu

TRAIN_DIR = "E:\\DataSets\\MAHDBase_TrainingSet\\TrainingSet.csv"
TEST_DIR = "E:\\DataSets\\MAHDBase_TrainingSet\\TestSet.csv"
LABELS_DIR = "E:\\DataSets\\MAHDBase_TrainingSet\\TrainingLabelsOnehot.csv"
TEST_LABELS_DIR = "E:\\DataSets\\MAHDBase_TrainingSet\\TestLabelsOnehot.csv"
MODEL_NAME = 'arabic_digit_classifier'
NUM_STEPS = 2000
BATCH_SIZE = 100
IMAGE_SIZE = 28
NUM_CLASSES = 10

X = np.genfromtxt (TRAIN_DIR, delimiter=',')
Y = np.genfromtxt(LABELS_DIR, delimiter=',')
X_test = np.genfromtxt(TEST_DIR, delimiter=',')
Y_test = np.genfromtxt(TEST_LABELS_DIR, delimiter=',')

def model_input(input_node_name, keep_prob_node_name):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE *IMAGE_SIZE,1], name=input_node_name)
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
    return x, keep_prob, y_


def build_model(x, keep_prob, y_, output_node_name):
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    # 28*28*1 // [-1,28,28,1]-->[batch, in_height, in_width, in_channels] : Batch is a dimension that
    # allows you to have a collection of images. This order is called NHWC. The other
    #option is NCWH

    conv1 = tf.layers.conv2d(x_image, 64, 3, 1, 'same', activation=tf.nn.relu)
    # 28*28*64
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'same')
    # 14*14*64

    conv2 = tf.layers.conv2d(pool1, 128, 3, 1, 'same', activation=tf.nn.relu)
    # 14*14*128
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'same')
    # 7*7*128

    conv3 = tf.layers.conv2d(pool2, 256, 3, 1, 'same', activation=tf.nn.relu)
    # 7*7*256
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, 'same')
    # 4*4*256

    flatten = tf.reshape(pool3, [-1, 4 * 4 * 256])
    fc = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
    dropout = tf.nn.dropout(fc, keep_prob)
    logits = tf.layers.dense(dropout, 10)
    outputs = tf.nn.softmax(logits, name=output_node_name)

    # loss

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    # train step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    return train_step, loss, accuracy, merged_summary_op


def train(x, keep_prob, y_, train_step, loss, accuracy,
          merged_summary_op, saver):
    print("training start...")


    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        tf.train.write_graph(sess.graph_def, 'out',
                             MODEL_NAME + '.pbtxt', True)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('logs/',
                                               graph=tf.get_default_graph())

        for step in range(NUM_STEPS):
            no_itr_per_epoch = len(X)//BATCH_SIZE
            previous_batch = 0
            for i in range(no_itr_per_epoch):
                current_batch = previous_batch + BATCH_SIZE
                x_input = X[previous_batch:current_batch]
                x_images = np.reshape(x_input, [BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE, 1])
                np.expand_dims(x_images, axis=0)
                y_input = Y [previous_batch:current_batch]
                y_label = np.reshape(y_input, [BATCH_SIZE, NUM_CLASSES])
                np.expand_dims(y_label, axis=0)
                previous_batch = previous_batch + BATCH_SIZE
                if step % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: x_images, y_: y_label, keep_prob: 1.0})
                    print('step %d, training accuracy %f' % (step, train_accuracy))
                    _, summary = sess.run([train_step, merged_summary_op], feed_dict={x: x_images, y_: y_label, keep_prob: 0.5})
            summary_writer.add_summary(summary, step)

        saver.save(sess, 'out/' + MODEL_NAME + '.chkp')
        x_test = np.reshape(X_test ,[len(X_test), IMAGE_SIZE*IMAGE_SIZE, 1])
        np.expand_dims(x_test, axis=0)

        # y_test = np.reshape(Y_test, )
        test_accuracy = accuracy.eval(feed_dict={x: x_test,
                                                 y_: Y_test,
                                                 keep_prob: 1.0})
        print('test accuracy %g' % test_accuracy)

    print("training finished!")


def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
                              'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
                              "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)



    print("graph saved!")
    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

def main():
    if not path.exists('out'):
        os.mkdir('out')

    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'

    x, keep_prob, y_ = model_input(input_node_name, keep_prob_node_name)

    train_step, loss, accuracy, merged_summary_op = build_model(x, keep_prob,
                                                                y_, output_node_name)
    saver = tf.train.Saver()

    train(x, keep_prob, y_, train_step, loss, accuracy,
          merged_summary_op, saver)

    export_model([input_node_name, keep_prob_node_name], output_node_name)


if __name__ == '__main__':
    main()


