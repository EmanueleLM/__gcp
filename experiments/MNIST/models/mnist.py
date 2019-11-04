# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:21:07 2019

@author: Emanuele

Simple neural network to classify images in MNIST dataset.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def _normalize(x, std):
    
    import tensorflow as tf

    def py_func_init(out):
        import numpy as np
        shape = out.shape
        out = np.reshape(out, [-1, shape[-1]])
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        out = np.reshape(out, shape)
        return out

    return x.assign(tf.py_func(py_func_init, [x], tf.float32))


def conv(x, kernel_size, num_outputs, name, act, stride=1, padding="SAME", bias=True, std=1.0):
    
    import tensorflow as tf

    w = tf.get_variable(name + "/w", [kernel_size, kernel_size, x.get_shape()[-1], num_outputs])

    w.reinitialize = _normalize(w, std=std)

    ret = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=padding)
    if bias:
        b = tf.get_variable(name + "/b", [1, 1, 1, num_outputs], initializer=tf.zeros_initializer)

        b.reinitialize = b.assign(tf.zeros_like(b))

        return act(ret + b)
    else:
        return act(ret)
    

def dense(x, size, name, act, weight_init=None, bias=True, std=1.0):
    
    import tensorflow as tf

    w = tf.get_variable(name + "/w", [x.get_shape()[1], size])

    w.reinitialize = _normalize(w, std=std)

    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer)

        b.reinitialize = b.assign(tf.zeros_like(b))

        return act(ret + b)
    else:
        return act(ret)
    
    
def flattenallbut0(x):
    import numpy as np
    return tf.reshape(x, [-1, int(np.prod(x.get_shape().as_list()[1:]))])


def MNIST_model(steps_number=1000, 
                batch_size=100, 
                save_to_file=False, 
                dst='', 
                get_m_info=False,
                get_nodes_strengths_m_info=False):
    
    import numpy as np
    import tensorflow as tf
    
    if get_m_info is True:
        import sys
        sys.path.append('../')
        import mutual_info as m_info
    
    # Read data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    image_size = 28
    labels_size = 10
    learning_rate = 0.05
    
    tf.reset_default_graph()
    
    # Define placeholders
    training_data = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
    labels = tf.placeholder(tf.float32, [None, labels_size])
    
    # Build the network (only output layer)
    l1 = conv(training_data, name='conv1', act=tf.nn.relu, num_outputs=32, kernel_size=8, stride=4, std=1.0)
    l2 = conv(l1, name='conv2', act=tf.nn.relu, num_outputs=32, kernel_size=4, stride=2, std=1.0)
    l2_f = flattenallbut0(l2)
    l3 = dense(l2_f, 256, 'fc1',  act=tf.nn.relu, weight_init=None, std=1.0)
    output = dense(l3, 10, 'fc2',  act=tf.nn.relu, weight_init=None, std=1.0)
    
    # dictionary used to estimate mutual info and the information plane trajectories
    layers_to_estimate = [l1, l2, l3, output]  # should add the input
    # I(x,t)
    I_x_t = {'i-l1': np.array([]), 'l1-l2': np.array([]), 
             'l2-l3': np.array([]), 'l3-o': np.array([])}
    # I(t,y)
    I_t_y = {'i-o': np.array([]), 'l1-o': np.array([]), 
             'l2-o': np.array([]), 'l3-o': np.array([])}
    tf_shapes = {'i': (batch_size, 28*28), 
                 'l1': (batch_size, np.prod(l1.get_shape().as_list()[1:])),
                 'l2': (batch_size, np.prod(l2.get_shape().as_list()[1:])),
                 'l3': (batch_size, np.prod(l3.get_shape().as_list()[1:])),
                 'o': (batch_size, np.prod(output.get_shape().as_list()[1:]))}
    
    # Define the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
    
    # Training step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # Accuracy calculation
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Run the training
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    
    for i in range(steps_number):
      # Get the next batch
      input_batch, labels_batch = mnist.train.next_batch(batch_size)
      feed_dict = {training_data: input_batch.reshape(batch_size, 28, 28, 1), labels: labels_batch}
      
      # append each activation to a separate vector then estimate I(li, lj)
      # we need to pickup few random values from each layer, otherwise the calculation becomes prohibitive
      if get_m_info is True:
          tmp = []
          mi_limit_sample = 100
          layers_indices = {}
          for key in tf_shapes.keys():
              layers_indices[key] = np.random.randint(0, tf_shapes[key][1], size=mi_limit_sample)
          for l in layers_to_estimate:
              tmp.append(sess.run(l, {training_data: input_batch.reshape(batch_size, 28, 28, 1), labels: labels_batch}))
          for key in I_x_t.keys():
              if key == 'i-l1':
                  ixt = m_info.mutual_information((np.take(input_batch.reshape(*(tf_shapes['i'])), axis=1, indices=layers_indices[key.split('-')[0]]),
                                                   np.take(tmp[0].reshape(*(tf_shapes['l1'])), axis=1, indices=layers_indices[key.split('-')[1]])))
                  ity = m_info.mutual_information((np.take(input_batch.reshape(*(tf_shapes['i'])), axis=1, indices=layers_indices[key.split('-')[0]]),
                                                   np.take(tmp[3].reshape(*(tf_shapes['o'])), axis=1, indices=layers_indices['o'])))
              elif key == 'l1-l2':
                  ixt = m_info.mutual_information((np.take(tmp[0].reshape(*(tf_shapes['l1'])), axis=1, indices=layers_indices[key.split('-')[0]]),
                                                   np.take(tmp[1].reshape(*(tf_shapes['l2'])), axis=1, indices=layers_indices[key.split('-')[1]])))
                  ity = m_info.mutual_information((np.take(tmp[0].reshape(*(tf_shapes['l1'])), axis=1, indices=layers_indices[key.split('-')[0]]),
                                                   np.take(tmp[3].reshape(*(tf_shapes['o'])), axis=1, indices=layers_indices['o'])))
              elif key == 'l2-l3':
                  ixt = m_info.mutual_information((np.take(tmp[1].reshape(*(tf_shapes['l2'])), axis=1, indices=layers_indices[key.split('-')[0]]),
                                                   np.take(tmp[2].reshape(*(tf_shapes['l3'])), axis=1, indices=layers_indices[key.split('-')[1]])))
                  ity = m_info.mutual_information((np.take(tmp[1].reshape(*(tf_shapes['l2'])), axis=1, indices=layers_indices[key.split('-')[0]]),
                                                   np.take(tmp[3].reshape(*(tf_shapes['o'])), axis=1, indices=layers_indices['o'])))
              elif key == 'l3-o':
                  ixt = m_info.mutual_information((np.take(tmp[2].reshape(*(tf_shapes['l3'])), axis=1, indices=layers_indices[key.split('-')[1]]),
                                                   np.take(tmp[3].reshape(*(tf_shapes['o'])), axis=1, indices=layers_indices[key.split('-')[1]])))
                  ity = m_info.mutual_information((np.take(tmp[2].reshape(*(tf_shapes['l3'])), axis=1, indices=layers_indices[key.split('-')[1]]),
                                                   np.take(tmp[3].reshape(*(tf_shapes['o'])), axis=1, indices=layers_indices['o'])))
              
              # save partial values of I(x,t) and I(t,y)
              I_x_t[key] = np.append(I_x_t[key], ixt)
              I_t_y[key.split('-')[0] + '-o'] = np.append(I_t_y[key.split('-')[0] + '-o'], ity)
    
      # Run the training step
      train_step.run(feed_dict=feed_dict)
      
      # Save the weights at the first and last iteration
      if save_to_file == True:
          if i == 0:
              initial_params = []
              for t_var in tf.trainable_variables():
                  initial_params.append(sess.run(t_var))
              print("[CUSTOM-LOGGER]: Saving initial params to file at relative path {}.".format(dst))
              np.save(dst + 'init_params.npy', np.asarray(initial_params))
          if i == steps_number-1:
              final_params = []
              for t_var in tf.trainable_variables():
                  final_params.append(sess.run(t_var))
              print("[CUSTOM-LOGGER]: Saving initial params to file at relative path {}.".format(dst))                  
              np.save(dst + 'fin_params.npy', np.asarray(final_params))
              
      # save all the weights from first to last iteration, then calculate the mutual info
      #  between nodes' strengths: save every <n> steps
      if get_nodes_strengths_m_info == True and i%5:
          initial_params = []
          for t_var in tf.trainable_variables():
              initial_params.append(sess.run(t_var))          
          #print("[CUSTOM-LOGGER]: Saving params for nodes strengths calculation to file at relative path {}.".format(dst))
          np.save(dst + 'params_s_minfo_gen_'+str(int(i/5))+'.npy', np.asarray(initial_params))
              
      # Print the accuracy progress on the batch every 100 steps
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print("[CUSTOM-LOGGER]: Step %d, training batch accuracy %g %%"%(i, train_accuracy*100))
    
    # Evaluate on the test set
    test_accuracy = accuracy.eval(feed_dict={training_data: mnist.test.images.reshape(10000, 28, 28, 1),
                                             labels: mnist.test.labels})
    print("[CUSTOM-LOGGER]: Test accuracy: %g %%"%(test_accuracy*100))
    
    if get_m_info is True:
        print("[CUSTOM-LOGGER]: Saving I(x,t) to file.")
        np.save('../results/info_theory/I_x_t.npy', I_x_t)
        print("[CUSTOM-LOGGER]: Saving I(t,y) to file.")
        np.save('../results/info_theory/I_t_y.npy', I_t_y)
