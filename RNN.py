import tensorflow as tf

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

dir_path = os.getcwd()

#load embendings

embeddings = np.loadtxt(os.path.join(dir_path, 'cache', 'embeddings.vec'))
embeddingsInstr = np.loadtxt(os.path.join(dir_path, 'cache', 'embeddingsInstr.vec'))

pickle_in = open('cache/database.dict', 'rb')
db = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('cache/traindata.list', 'rb')
traindata = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('cache/data2onehot.dict', 'rb')
dictionary = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('cache/data2onehotInstr.dict', 'rb')
dictionaryInstr = pickle.load(pickle_in)
pickle_in.close()

# seperate data by length and by each label in one class 'Y_labels'
# and by jump instruction 'Y_if'
#
from JavaClassParser import ByteCode
reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
X_train=[]
Y_train=[]
X_train_long=[]
Y_train_long=[]
X_labels = []
Y_labels = []

X_if = []
Y_if = []

for dclass in db.values():
    for method in dclass.values():
        instructions = method['x']
        labels = method['y']
        byteIndex = method['index']
        
        
        #
        #seperating by labels

        cur_section = []
        cur_label = labels[0]
        if len(instructions)<100:
            X_train.append(instructions)
            Y_train.append(labels)
        else:
            X_train_long.append(instructions)
            Y_train_long.append(labels)
        for instruction, label in zip(instructions,labels):
            if label != cur_label:
                X_labels.append(cur_section)
                Y_labels.append(cur_label)
                cur_section = []
                cur_label = label
            cur_section.append(dictionary.get(instruction,0))
        
        #print(instructions)
                        
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train_long = np.array(X_train_long)
Y_train_long = np.array(Y_train_long)
X_labels = np.array(X_labels)
Y_labels = np.array(Y_labels)

output_size = np.max(Y_labels[:,1])+1
learning_rate = 1e-1
batch_size = 50
hidden_size = 300
'''
num_features depends how do u we want to represent data
we can use w2v embbeding to send dense represetation
#num_features =embeddings.shape[1]
we can use sparse representation which for this example requires over 5000 long one hot vector
we can use sparse representation taking only instruction without arguments lowering one hot to 203 dim

in this simple model dense representation didnt show better result then filtered representation 
with only instructions
'''

#num_features = len(ByteCode.mnemonicMap)
num_features =embeddings.shape[1]
num_epochs = 20

sequence_length = 100

rnn2_graph = tf.Graph()

with rnn2_graph.as_default():
    
    sequence = tf.placeholder(tf.float64,[batch_size, sequence_length ,num_features])
    labels= tf.placeholder(tf.float64,[batch_size, sequence_length, output_size])
    seq_len = tf.placeholder(tf.int64, [batch_size])
    mask = tf.placeholder(tf.float64, [batch_size, sequence_length, output_size])

    cells_fw = [
        tf.nn.rnn_cell.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(hidden_size,activation=tf.nn.tanh),
        output_keep_prob = 0.8)
        for _ in range(5)]
    
    cells_bw = [
        tf.nn.rnn_cell.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(hidden_size,activation=tf.nn.tanh),
        output_keep_prob = 0.8)
        for _ in range(5)]
    

    
    W_1 = tf.Variable(tf.random_normal([1,hidden_size*2, hidden_size*2], dtype=tf.float64))
    b_1 = tf.Variable(tf.random_normal([hidden_size*2],dtype=tf.float64))
    W_2 = tf.Variable(tf.random_normal([1,hidden_size*2, output_size], dtype=tf.float64))
    b_2 = tf.Variable(tf.random_normal([output_size], dtype=tf.float64))

    #initial_state = cell_fw.zero_state(batch_size, dtype=tf.float64)


    #outputs, state = tf.nn.dynamic_rnn(cell, sequence, 
    #        initial_state=initial_state, sequence_length=seq_len)

    
    outputs, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw = cells_fw,
        cells_bw=cells_bw,
        dtype = tf.float64,
        sequence_length = seq_len,
        inputs = sequence)
    
    outputs = tf.concat(outputs, 2)
    
    outputs_2 = tf.nn.relu(tf.matmul(outputs, tf.tile(W_1, [tf.shape(outputs)[0],1,1])) + b_1)
    logits  = tf.matmul(outputs_2, tf.tile(W_2, [tf.shape(outputs_2)[0],1,1])) + b_2
    logits = logits*mask
    '''
    for output_batch, label_batch in zip(tf.unstack(outputs, axis=1), tf.unstack(labels, axis=1)):
        for output, label in zip(tf.unstack(output_batch, axis=0), tf.unstack(label_batch, axis=0)):
            output = tf.expand_dims(output, 0)
            label = tf.expand_dims(label, 0)
            logits = tf.matmul(output, W)+ b
            loss += tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
            correct_predictions += tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), tf.float64))
        #incorrect_prediciton += batch_size - tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)),tf.float64))
    '''
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits = logits)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #accuracy = correct_predictions/tf.reduce_sum(tf.cast(seq_len,tf.float64))
    #acc_val, acc_op = tf.metrics.accuracy(tf.argmax(labels,axis=2), tf.argmax(logits, axis=2))
    diff = batch_size*sequence_length - tf.reduce_sum(tf.cast(tf.equal(tf.argmax(labels, axis =2), tf.argmax(logits, axis=2)), tf.float32))
    nonpadsum = tf.cast(tf.reduce_sum(seq_len),tf.float32)
    accuracy = (nonpadsum- diff)/nonpadsum
    #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  




N = X_train.shape[0]
n_steps = N//batch_size

with tf.Session(graph=rnn2_graph) as session:
    iop = tf.global_variables_initializer()
    loc = tf.local_variables_initializer()
    session.run(iop)
    #session.run(loc)
    for epoch in range(num_epochs):
        acc_sum=0
        for step in range(n_steps):
            batch = X_train[batch_size*step:batch_size*(step+1)]
            sequence_len = np.array(list(map(lambda x : len(x), batch)))
            max_len = np.max(sequence_len)
            if max_len>sequence_length or (len(batch)<batch_size):
                continue
            data = np.zeros([batch_size, sequence_length, num_features])
            filled_labels = np.zeros([batch_size, sequence_length, output_size])
            filled_labels[:,:,0] = 0
            mask_val = np.zeros([batch_size, sequence_length, output_size])
            for i, seq in enumerate(Y_train[batch_size*step:batch_size*(step+1)]):
                for j, label in enumerate(seq):
                    onehot=np.zeros(output_size)
                    onehot[label[1]] = 1
                    filled_labels[i,j,:] = onehot
                    mask_val[i,j,:] = 1
            for i in range(batch_size):
                for j in range(len(batch[i])):
                    data[i,j,:] = embeddings[dictionary.get(batch[i][j],0)]
            
            feed = {sequence:data, labels:filled_labels, seq_len:sequence_len, mask:mask_val}
            
            logits_val, diff_val, acc_fixed, _= session.run([logits, diff, accuracy, train_step], feed_dict=feed)
            
            #acc_fixed = session.run([accuracy], feed_dict={seq_len:sequence_len})
            #session.run(loc)
            
            #print(acc_fixed)
            #print(diff_val)
            #print(np.sum(filled_labels))
            #print(np.sum(sequence_len))
            #print(np.argmax(filled_labels,axis=2)[0][:sequence_len[0]])
            #print(np.argmax(logits_val,axis=2)[0][:sequence_len[0]])
            '''
            zum = 0
            for i,sl in enumerate(sequence_len):
                zum += np.sum(np.argmax(logits_val[i,sl:,:],axis=1))
            print(zum)
            '''
            #print()
            acc_sum += acc_fixed
            
        print("Epoch %d, step %d ,acc: %f" % (epoch, step, acc_sum/n_steps))
        index = np.argmax(sequence_len)
        print(np.argmax(filled_labels,axis=2)[index][:sequence_len[index]])
        print(np.argmax(logits_val,axis=2)[index][:sequence_len[index]])
