#coding:utf8
'''
Created on Mar 6, 2017

@author: lab
'''

from sklearn.metrics import mean_squared_error
# from sklearn.manifold import TSNE
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def import_data():
    #import data
    dictionary_data = np.load("/media/lab/bangong/extract data Rodrigo/Rodrigo/laoshi_data/multitask_ball_mill/data.npy")
    dictionary_data = dictionary_data.item()
    dictionary_data = make_semisupervise_data(dictionary_data)
    return dictionary_data
def make_semisupervise_data(dictionary_data):
    
    dictionary_data2 = {"train0_0":dictionary_data["train0_0"], 
                        "train0_1":dictionary_data["train0_1"], 
                        "train0_2":dictionary_data["train0_2"],
                        "train0_3":dictionary_data["train0_3"],
                        "train0_4":dictionary_data["train0_4"],}
    train_file = list("train0_0")
    
    n_lables = 30
    for i in range(1,8):
        train_file[-3]=str(i)
        for ii in range(0, 5):
            train_file[-1]=str(ii)
            data_x = dictionary_data["".join(train_file)]
            steps_size=(data_x.shape[0]/14)/n_lables
            data_ = np.zeros((0, data_x.shape[1]))
            for p in range(0, data_x.shape[0], 14*steps_size):
                data_ = np.append(data_, data_x[p:p+7, :], axis=0)
            
            dictionary_data2["".join(train_file)]=data_
    test_file = list("test0_0")
    
    for i in range(0,8):
        test_file[-3]=str(i)
        for ii in range(0, 5):
            test_file[-1]=str(ii)
            dictionary_data2["".join(test_file)] = dictionary_data["".join(test_file)]
    
    return dictionary_data2
            
            
#import data
#train data
data = import_data()

#hiperparametros
hidden1 = 500
hidden2 = 250
lstm_hidden = hidden2
out_size=1
lr = 0.0015
n_epochs = 10000
displayer = 2
batch_size = 250
drop=0.7
n_steps = 6
ventana = 14
n_input = data["train0_0"].shape[1]
n_output = 1
batch_init = 0
batch_init_test = 0

def sequencesFromTrainingData(batch_size, n_steps, data_n): 
    global batch_init
    
    name_data = list("train0_0")
    name_data[-3] = str(data_n)
    train=[]
    for i in range(5):
        name_data[-1]=str(i)
        duplicate_data = np.append(data["".join(name_data)], data["".join(name_data)], axis=0)
        duplicate_data = np.append(duplicate_data, duplicate_data, axis = 0)
        train.append(duplicate_data)
    
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    if int(name_data[-3])==0:
        batch_master = batch_size
        iner_steps = ventana-n_steps
    else:
        batch_master = train[0].shape[0]/(n_steps+1)
        iner_steps = 1
    
    data_x = np.split(train[0][batch_init:batch_init+ventana*batch_master, :], batch_master, axis=0)
    data_y1 = np.split(train[1][batch_init:batch_init+ventana*batch_master, :], batch_master, axis=0)
    data_y2 = np.split(train[2][batch_init:batch_init+ventana*batch_master, :], batch_master, axis=0)
    data_y3 = np.split(train[3][batch_init:batch_init+ventana*batch_master, :], batch_master, axis=0)
    data_y4 = np.split(train[4][batch_init:batch_init+ventana*batch_master, :], batch_master, axis=0)
    for p in range(batch_master):
        sample = data_x[p]
        lavel1 = data_y1[p]
        lavel2 = data_y2[p]
        lavel3 = data_y3[p]
        lavel4 = data_y4[p]
        for i in range(iner_steps):
            X.append(sample[i:i+n_steps,:])
            Y1.append(lavel1[i+n_steps,:])
            Y2.append(lavel2[i+n_steps,:])
            Y3.append(lavel3[i+n_steps,:])
            Y4.append(lavel4[i+n_steps,:])
    batch_init = batch_init + ventana*batch_master 
    if (batch_init+ventana*batch_master)>=train[0].shape[0]:
        batch_init = 0
    
    return np.asarray(X), np.asarray(Y1), np.asarray(Y2), np.asarray(Y3), np.asarray(Y4)


def sequencesFromTestData(n_steps, data_n): 
    global batch_init_test
    name_data = list("test0_0")
    name_data[-3] = str(data_n)
    test=[]
    for i in range(5):
        name_data[-1]=str(i)
        duplicate_data = np.append(data["".join(name_data)], data["".join(name_data)], axis=0)
        test.append(duplicate_data)
    
    
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    batch_size = test[0].shape[0]/ventana
    data_x = np.split(test[0], batch_size, axis=0)
    data_y1 = np.split(test[1], batch_size, axis=0)
    data_y2 = np.split(test[2], batch_size, axis=0)
    data_y3 = np.split(test[3], batch_size, axis=0)
    data_y4 = np.split(test[4], batch_size, axis=0)
    for p in range(batch_size):
        sample = data_x[p]
        lavel1 = data_y1[p]
        lavel2 = data_y2[p]
        lavel3 = data_y3[p]
        lavel4 = data_y4[p]
        for i in range(ventana-n_steps):
            X.append(sample[i:i+n_steps,:])
            Y1.append(lavel1[i+n_steps,:])
            Y2.append(lavel2[i+n_steps,:])
            Y3.append(lavel3[i+n_steps,:])
            Y4.append(lavel4[i+n_steps,:])

    
    return np.asarray(X), np.asarray(Y1), np.asarray(Y2), np.asarray(Y3), np.asarray(Y4)


def main():
    global batch_init 
    
    #set up tensorflow imputs 
    x = tf.placeholder("float", [None, n_steps, n_input])
    y1 = tf.placeholder("float", [None, 1])
    y2 = tf.placeholder("float", [None, 1])
    y3 = tf.placeholder("float", [None, 1])
    y4 = tf.placeholder("float", [None, 1])
    keep_prob = tf.placeholder("float")
    #saver
#     saver = tf.train.Saver()
    # reshape x
    x_internal = tf.transpose(x,[1, 0, 2])
    x_internal = tf.reshape(x_internal, [-1, n_input]) # (n_steps*batch_size, n_input)
    #lstm cells
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden, forget_bias=1.0)
    istate = lstm_cell.zero_state(batch_size*(ventana-n_steps), tf.float32)
    #inicializate the weights and bias
    w0 = tf.Variable(tf.random_normal([n_input, hidden1]))
    b0 = tf.Variable(tf.random_normal([hidden1]))
    #output weights
    w_tk1 = tf.Variable(tf.random_normal([hidden2, out_size]))
    w_tk2 = tf.Variable(tf.random_normal([hidden2, out_size]))
    w_tk3 = tf.Variable(tf.random_normal([hidden2, out_size]))
    w_tk4 = tf.Variable(tf.random_normal([hidden2, out_size]))
    #output bias
    b_tk1 = tf.Variable(tf.random_normal([out_size]))
    b_tk2 = tf.Variable(tf.random_normal([out_size]))
    b_tk3 = tf.Variable(tf.random_normal([out_size]))
    b_tk4 = tf.Variable(tf.random_normal([out_size]))
    #graph
    layer1 = tf.nn.relu(tf.add(tf.matmul(x_internal, w0), b0))
    layer1 = tf.nn.dropout(layer1, keep_prob)
    layer1 = tf.split(0, n_steps, layer1) # n_steps * (batch_size, n_hidden)
    outputs, states = tf.nn.rnn(lstm_cell, layer1, dtype=tf.float32)
    #outputs
    out_tk1 = tf.add(tf.matmul(outputs[-1], w_tk1), b_tk1)
    out_tk2 = tf.add(tf.matmul(outputs[-1], w_tk2), b_tk2)
    out_tk3 = tf.add(tf.matmul(outputs[-1], w_tk3), b_tk3)
    out_tk4 = tf.add(tf.matmul(outputs[-1], w_tk4), b_tk4)
    #loss function
    loss1 = tf.reduce_mean(tf.square(tf.add(out_tk1, -y1)))
    loss2 = tf.reduce_mean(tf.square(tf.add(out_tk2, -y2)))
    loss3 = tf.reduce_mean(tf.square(tf.add(out_tk3, -y3)))
    loss4 = tf.reduce_mean(tf.square(tf.add(out_tk4, -y4)))
    #the final loss
    loss = tf.add(tf.add(loss1, loss2), tf.add(loss3, loss4))
    #optimizer
    main_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
#     optimizer1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss1)
#     optimizer2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss2)
#     optimizer3 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss3)
#     optimizer4 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss4)
    #graph init
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        error_y = 100
        iterations = 0
        while error_y > 0.5:
            batch = sequencesFromTrainingData(batch_size, n_steps, data_n=0)
            feed_dict ={x: batch[0], y1: batch[1], y2: batch[2].dot(100), y3: batch[3].dot(100), y4: batch[4], keep_prob: drop}
            sess.run(main_optimizer, feed_dict=feed_dict)
            
            if (iterations%displayer)==0:
                
                feed_dict ={x: batch[0], y1: batch[1], y2: batch[2].dot(100), y3: batch[3].dot(100), y4: batch[4], keep_prob: 1}
                t_loss = sess.run([loss, loss1, loss2, loss3, loss4], feed_dict=feed_dict)
                error_y = t_loss[4]
                print "General training loss: {0}, loss1: {1}, loss2: {2}, loss3: {3}, loss4: {4}".format(np.sqrt(t_loss[0]), np.sqrt(t_loss[1]), np.sqrt(t_loss[2]), np.sqrt(t_loss[3]), np.sqrt(t_loss[4]))
                batch = sequencesFromTestData(n_steps, data_n=0)
                feed_dict ={x: batch[0], y1: batch[1], y2: batch[2].dot(100), y3: batch[3].dot(100), y4: batch[4], keep_prob: 1}
                t_loss = sess.run([loss, loss1, loss2, loss3, loss4], feed_dict=feed_dict)
                print "General Test     loss: {0}, loss1: {1}, loss2: {2}, loss3: {3}, loss4: {4}".format(np.sqrt(t_loss[0]), np.sqrt(t_loss[1]), np.sqrt(t_loss[2]), np.sqrt(t_loss[3]), np.sqrt(t_loss[4]))
            
            iterations +=1
        
        batch = sequencesFromTestData(n_steps, data_n=0)
        feed_dict ={x: batch[0], y1: batch[1], y2: batch[2].dot(100), y3: batch[3].dot(100), y4: batch[4], keep_prob: 1}
        final_loss = sess.run(loss4, feed_dict=feed_dict)
        print "General Test loss: {0}".format(final_loss)
        history_iter = []
        
        for num_task in range(1,8):
            history_iter.append(iterations)
            iterations = 0
            batch_init = 0
            error_y = final_loss*100
            
            while error_y>final_loss:    
                
                batch = sequencesFromTrainingData(batch_size, n_steps, data_n=num_task)
                feed_dict ={x: batch[0], y1: batch[1], y2: batch[2].dot(100), y3: batch[3].dot(100), y4: batch[4], keep_prob: drop}
                sess.run(main_optimizer, feed_dict= feed_dict)
                
                if (iterations%displayer)==0:
                        
                    feed_dict ={x: batch[0], y1: batch[1], y2: batch[2].dot(100), y3: batch[3].dot(100), y4: batch[4], keep_prob: 1}
                    t_loss = sess.run([loss, loss1, loss2, loss3, loss4], feed_dict=feed_dict)
                    print "General training loss: {0}, loss1: {1}, loss2: {2}, loss3: {3}, loss4: {4}".format(np.sqrt(t_loss[0]), np.sqrt(t_loss[1]), np.sqrt(t_loss[2]), np.sqrt(t_loss[3]), np.sqrt(t_loss[4]))
                    batch = sequencesFromTestData(n_steps, data_n=num_task)
                    feed_dict ={x: batch[0], y1: batch[1], y2: batch[2].dot(100), y3: batch[3].dot(100), y4: batch[4], keep_prob: 1}
                    t_loss = sess.run([loss, loss1, loss2, loss3, loss4], feed_dict=feed_dict)
                    error_y = t_loss[4]
                    print "General Test loss:     {0}, loss1: {1}, loss2: {2}, loss3: {3}, loss4: {4}".format(np.sqrt(t_loss[0]), np.sqrt(t_loss[1]), np.sqrt(t_loss[2]), np.sqrt(t_loss[3]), np.sqrt(t_loss[4]))
                iterations += 1
                
            batch = sequencesFromTestData(n_steps, data_n=num_task)
            feed_dict ={x: batch[0], y1: batch[1], y2: batch[2].dot(100), y3: batch[3].dot(100), y4: batch[4], keep_prob: 1}
            out1, out2, out3, out4 = sess.run([out_tk1, out_tk2, out_tk3, out_tk4], feed_dict={x: batch[0], keep_prob: 1})
            print "error output 1: {0}, error output 2:, {1}, error output 3: {2}, error output 4: {3}".format(mean_squared_error(out1, batch[1])**0.5, 
                                                                                                        mean_squared_error(out2, batch[2].dot(100))**0.5, 
                                                                                                        mean_squared_error(out3, batch[3].dot(100))**0.5, 
                                                                                                        mean_squared_error(out4, batch[4])**0.5)
            _, ax = plt.subplots(2,2)
            ax[0,0].plot(out1[:out1.shape[0]/2,:], "--r", batch[1][:batch[1].shape[0]/2,:], "b")
            ax[0,1].plot(out2.dot(0.01)[:out2.shape[0]/2,:], "--r", batch[2][:batch[2].shape[0]/2,:], "b")
            ax[1,0].plot(out3.dot(0.01)[:out3.shape[0]/2,:], "--r", batch[3][:batch[3].shape[0]/2,:], "b")
            ax[1,1].plot(out4[:out4.shape[0]/2,:], "--r", batch[4][:batch[4].shape[0]/2,:], "b")
    
if __name__=="__main__":
    main()