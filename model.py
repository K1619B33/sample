from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'
floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper
import numpy as np

batch_id = 3
sample_id = 999
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

def normalize(x):
    
    norm =np.array(x/255)
    return norm


tests.test_normalize(normalize)

def one_hot_encode(x, n_values=10):  
    
    from sklearn.preprocessing import OneHotEncoder
    
    encoder = OneHotEncoder(n_values = n_values)
    one_hot_encoded_labels = encoder.fit_transform(np.array(x).reshape(-1,1)).toarray()
    return one_hot_encoded_labels


tests.test_one_hot_encode(one_hot_encode)


helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


import pickle
import problem_unittests as tests
import helper

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

import tensorflow as tf

def neural_net_image_input(image_shape):
    
    return tf.placeholder(tf.float32, shape=(None, *image_shape), name='x')


def neural_net_label_input(n_classes):
    
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')


def neural_net_keep_prob_input():
    
    return tf.placeholder(tf.float32, shape=(None), name='keep_prob')

tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    
    
    depth = x_tensor.get_shape().as_list()[3]
    weight = tf.Variable(tf.truncated_normal([*conv_ksize,depth,conv_num_outputs], mean=.0, stddev=.01))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    padding = 'SAME'
    
    conv = tf.nn.conv2d(x_tensor, weight, padding=padding, strides=[1,*conv_strides,1], use_cudnn_on_gpu=True)
    conv = tf.nn.bias_add(conv, bias)
    conv = tf.nn.relu(conv)
    
    return tf.nn.max_pool(value=conv, ksize=[1,*pool_ksize,1], strides=[1,*pool_strides,1], padding=padding, name='max_pool')


tests.test_con_pool(conv2d_maxpool)

def flatten(x_tensor):
   
    
    return tf.contrib.layers.flatten(x_tensor)
    
   


tests.test_flatten(flatten)

def fully_conn(x_tensor, num_outputs):
    
    batch_size = x_tensor.get_shape().as_list()[-1]
    weights = tf.Variable(tf.truncated_normal([batch_size, num_outputs], mean=.0, stddev=.01), name="weights_fully_conn")
    bias = tf.Variable(tf.zeros([num_outputs]), name="bias_fully_conn")
    linear = tf.nn.bias_add(tf.matmul(x_tensor, weights), bias)
    return tf.nn.relu(linear)


tests.test_fully_conn(fully_conn)

def output(x_tensor, num_outputs):
    
    
    depth = x_tensor.get_shape().as_list()[-1]
    weights = tf.Variable(tf.random_normal([depth, num_outputs], mean=.0, stddev=.01), name="weights_fully_conn")
    bias = tf.Variable(tf.zeros([num_outputs]))
    output_layer = tf.nn.bias_add(tf.matmul(x_tensor, weights), bias)
    return output_layer


tests.test_output(output)

def conv_net(x, keep_prob):
    
    
    conv_num_outputs = [64,128,256]

    conv_strides = [(2,2), (2,2), (2,2)]
    conv_ksize = [(4,4), (6,6), (8,8)]

    pool_ksize = (4,4)
    pool_strides = (2,2)
    
    conv_layer_1 = conv2d_maxpool(x, conv_num_outputs[0], conv_ksize[0], conv_strides[0], pool_ksize, pool_strides)

    conv_layer_2 = conv2d_maxpool(conv_layer_1,conv_num_outputs[1], conv_ksize[1], conv_strides[1], pool_ksize, pool_strides)

    conv_layer_2 = tf.nn.dropout(conv_layer_2, keep_prob)

    
    
    x_tensor = flatten(conv_layer_2)

   
    num_outputs = (256, 128)
    fully_conn_1 = fully_conn(x_tensor, num_outputs[0])
    fully_conn_1 = tf.nn.dropout(fully_conn_1, keep_prob)
    fully_conn_2 = fully_conn(fully_conn_1, num_outputs[1])

   
    
    
    final_out = output(fully_conn_2, 10)
    
   
    return final_out



tf.reset_default_graph()


x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()


logits = conv_net(x, keep_prob)


logits = tf.identity(logits, name='logits')


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)


correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    
   
    session.run(optimizer, feed_dict = {x: feature_batch, 
                                       y: label_batch, 
                                       keep_prob: keep_probability})



tests.test_train_nn(train_neural_network)

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    
    cost = sess.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.})
    
    acc = sess.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.})
    
    print('Loss: ', cost, ' Accuracy: ', acc)
    
    pass


epochs = 45
batch_size = 512
keep_probability = 0.7


print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)


save_model_path = r'C:\Users\Feroz\Downloads\Compressed\udacity-project-image-classification-master/image_classification'

print('Training...')
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    
    for epoch in range(epochs):
        
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)


%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import tensorflow as tf
import pickle
import helper
import random


try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = r'C:\Users\Feroz\Downloads\Compressed\udacity-project-image-classification-master\image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
   

    test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

       
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
       
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

       
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
