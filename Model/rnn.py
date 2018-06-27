import json
import os.path

import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import tensorflow as tf
from Model import logger
from Model.utils import *

class RNN(object):
    """Recursive Neural Network"""

    @classmethod
    def restore(cls, session, model_directory):
        """
        Restore a previously trained model
        :param session: session into which to restore the model
        :type session: TensorFlow Session
        :param model_directory: directory to which the model was saved
        :type model_directory: str
        :return: trained model
        :rtype: RNN
        """
        with open(cls._parameters_file(model_directory)) as f:
            parameters = json.load(f)
        
        additonal_param = Additional_Parameters(
            args_dim = parameters["args_dim"], 
            bidirectional=parameters["bidirectional"],
            nonlinear=parameters["nonlinear"],
            encode_int = parameters.get("encode_int", True))
        model = cls(parameters["max_gradient"],
                    parameters["batch_size"], parameters["time_steps"], parameters["vocabulary_size"],
                    parameters["hidden_units"], parameters["output_size"], parameters["layers"],
                    additional_parameters=additonal_param)
        model.acc = parameters["best_valid_acc"]

        tf.train.Saver().restore(session, cls._model_file(model_directory))
        with open(cls._vocabulary_file(model_directory)) as f:
            model.vocabulary = json.load(f)
        return model

    @staticmethod
    def _parameters_file(model_directory):
        return os.path.join(model_directory, "parameters.json")

    @staticmethod
    def _model_file(model_directory):
        return os.path.join(model_directory, "model")
    
    @staticmethod
    def _vocabulary_file(model_directory):
        return os.path.join(model_directory, "vocabulary.json")

    def __init__(self, 
                 max_gradient,
                 batch_size,
                 time_steps,
                 vocabulary_size,
                 hidden_units,
                 output_size,
                 layers,
                 additional_parameters = Additional_Parameters()):
        
        self.max_gradient = max_gradient
        self.layers = layers
        self.additional_parameters = additional_parameters
        self.acc=0
        
        # Add vocabulary slots of out of vocabulary (index 1) and padding (index 0).
        # vocabulary_size += 2

        self.cost = 0
        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
        with tf.name_scope("Parameters"):
            self.learning_rate = tf.placeholder(tf.float64, name="learning_rate")
            self.keep_probability = tf.placeholder(tf.float64, name="keep_probability")

        with tf.name_scope("Input"):
            self.input = tf.placeholder(tf.int32, shape=(batch_size, time_steps), name="input")
            self.labels = tf.placeholder(tf.float64, shape=(batch_size, time_steps, output_size), name="targets")
            self.mask = tf.placeholder(tf.float64, shape=(batch_size, time_steps, output_size), name="mask")
            self.seq_len = tf.placeholder(tf.int32, [batch_size,])
        
        with tf.name_scope("Embedding"):
            embedding_size = hidden_units
            if self.additional_parameters.args_dim:
                embedding_size = hidden_units - self.additional_parameters.args_dim
            self.embedding = tf.get_variable("embedding_1", (vocabulary_size, embedding_size), 
                                            initializer = tf.contrib.layers.xavier_initializer(),
                                            dtype=tf.float64) 
            #self.embedding = initializer((vocabulary_size, hidden_units))
            self.embedded_input = tf.nn.embedding_lookup(self.embedding, self.input, name="embedded_input")

        self.input_rnn = self.embedded_input

        if self.additional_parameters.args_dim:
            with tf.name_scope("Argument_encoding"):
                self.input_args = tf.placeholder(tf.float64, shape=(batch_size, time_steps, self.additional_parameters.args_dim))
                self.input_rnn = tf.concat([self.embedded_input, self.input_args],2)
        
        if self.additional_parameters.encode_int:
            with tf.name_scope("Encoded_numeric_args"):
                self.encoded_int_args = tf.placeholder(tf.float64, shape=(batch_size, time_steps, 32), name="encoded_int")
                self.input_rnn = tf.concat([self.embedded_input, self.encoded_int_args], 2)

        if self.additional_parameters.bidirectional:
            with tf.name_scope("RNN_Bidirectional"):
                cell = lambda : tf.nn.rnn_cell.DropoutWrapper(
                                tf.contrib.rnn.BasicLSTMCell(hidden_units),
                                output_keep_prob = self.keep_probability)
                #tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_units, 
                #                dropout_keep_prob=self.keep_probability) 
                cells_fw = [
                    cell()                
                    for _ in range(layers)]
    
                cells_bw = [
                    cell()
                    for _ in range(layers)]
                
                self.state = ([cell.zero_state(batch_size, dtype=tf.float64) for cell in cells_fw],
                              [cell.zero_state(batch_size, dtype=tf.float64) for cell in cells_bw])
                outputs, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                cells_fw = cells_fw,
                                                cells_bw=cells_bw,
                                                #initial_states_fw = self.state[0],
                                                #initial_states_bw = self.state[1],
                                                sequence_length = self.seq_len,
                                                dtype = tf.float64,
                                                inputs = self.input_rnn)
                
                self.next_state = (states_fw, states_bw)
                #self.outputs_rnn = tf.concat(outputs, 2)
                self.outputs_rnn = outputs
    
            '''
            with tf.name_scope("RNN_attention"):
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(128, self.outputs_rnn_1)
                cells_fw_2 = [
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.contrib.seq2seq.AttentionWrapper(
                            tf.contrib.rnn.LSTMCell(hidden_units, use_peepholes=True),
                            attention_mechanism,
                            attention_layer_size=64
                        ),
                    output_keep_prob = self.keep_probability)
                    for _ in range(layers)]
    
                cells_bw_2 = [
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.contrib.seq2seq.AttentionWrapper(
                            tf.contrib.rnn.LSTMCell(hidden_units, use_peepholes=True),
                            attention_mechanism,
                            attention_layer_size=64
                        ),
                    output_keep_prob = self.keep_probability)
                    for _ in range(layers)]
                
                self.state_att = ([cell.zero_state(batch_size, dtype=tf.float64) for cell in cells_fw_2],
                              [cell.zero_state(batch_size, dtype=tf.float64) for cell in cells_bw_2])
                
                outputs_2, states_fw_2, states_bw_2 = tf.contrib.rnn.stack_bidirectional_rnn(
                                                cells_fw = cells_fw_2,
                                                cells_bw = cells_bw_2,
                                                initial_states_fw = self.state_att[0],
                                                initial_states_bw = self.state_att[1],
                                                sequence_length = self.seq_len,
                                                dtype = tf.float64,
                                                inputs = tf.unstack(self.input_rnn,axis=1))
                
                self.next_state_att = (states_fw_2, states_bw_2)
                self.outputs_rnn_att = outputs_2
            
            hidden_layer_size = hidden_units
            self.outputs_rnn = outputs_2
            '''
        else:
            with tf.name_scope("RNN"):
                cell = lambda:tf.nn.rnn_cell.DropoutWrapper(
                                tf.nn.rnn_cell.LSTMCell(hidden_units),
                                output_keep_prob=self.keep_probability)

                cells = [cell() for _ in range(layers)]
                rnn_layers = tf.nn.rnn_cell.MultiRNNCell(cells)
                self.state = rnn_layers.zero_state(batch_size, dtype=tf.float64)
                self.outputs_rnn, self.next_state = tf.nn.dynamic_rnn(rnn_layers, self.input_rnn,
                                                                sequence_length=self.seq_len,
                                                                initial_state=self.state)
        
        if self.additional_parameters.nonlinear:
            with tf.name_scope("Nonlinear_output"):
                hidden_layer_size = self.outputs_rnn.get_shape()[2]
                self.w_nl = tf.get_variable("w_nl", (1, hidden_layer_size, hidden_layer_size),
                                            initializer = tf.contrib.layers.xavier_initializer(),
                                            dtype=tf.float64)
                self.b_nl = tf.get_variable("b_nl", hidden_layer_size, initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
                self.outputs = tf.nn.relu(tf.matmul(self.outputs_rnn, tf.tile(self.w_nl, [tf.shape(self.outputs_rnn)[0],1,1])) + self.b_nl)
                self.cost += 0.01*tf.nn.l2_loss(self.w_nl)
        else:
            self.outputs = self.outputs_rnn
        with tf.name_scope("Cost"):
            #self.w = tf.Variable(initializer((hidden_size, output_size)), dtype = tf.float64, name = "W")
            #self.b = tf.Variable(initializer((output_size)), dtype = tf.float64, name = "b")
            # Concatenate all the batches into a single row.
            layer_size = self.outputs.get_shape()[2]
            self.flattened_outputs = tf.reshape(self.outputs, [-1, tf.shape(self.outputs)[2]],
                                                name="flattened_outputs")
            self.flattened_labels = tf.reshape(self.labels, [-1, output_size],
                                                name="flattened_labels")
            self.flattened_mask = tf.reshape(self.mask, [-1, output_size],
                                                name="flattened_mask")
            # Project the outputs onto the vocabulary.
            self.w = tf.get_variable("w_logits", (1,layer_size, output_size), initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            self.b = tf.get_variable("b_logits", output_size, initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            #self.predicted = tf.matmul(self.flattened_outputs, self.w) + self.b
            #self.predicted = tf.multiply(self.predicted, self.flattened_mask)
            self.predicted = tf.matmul(self.outputs, tf.tile(self.w, [tf.shape(self.outputs)[0],1,1])) + self.b
            self.predicted = self.predicted*self.mask
            #self.predicted = tf.reshape(self.predicted, shape=(batch_size, time_steps, output_size))
            # Compare predictions to labels.
            #self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.flattened_labels, logits=self.predicted, name="loss")
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predicted, labels=self.labels)
            self.predicted = tf.reshape(self.predicted, [-1, tf.shape(self.predicted)[2]])
            self.cost += tf.div(tf.reduce_sum(self.loss), batch_size, name="cost")
            #self.cost = self.loss
            batch_num = tf.div(tf.reduce_sum(self.mask), output_size) 
            padding_num = batch_size*time_steps - batch_num
            correct_num = tf.reduce_sum( tf.cast(tf.equal(tf.argmax(self.flattened_labels, axis=1), tf.argmax(self.predicted, axis=1)), tf.float64) ) - padding_num
            self.accuracy = correct_num/batch_num

        with tf.name_scope("Train"):
            self.validation_perplexity = tf.Variable(dtype=tf.float64, initial_value=float("inf"), trainable=False,
                                                     name="validation_perplexity")
            tf.summary.scalar(self.validation_perplexity.op.name, self.validation_perplexity)
            self.training_epoch_perplexity = tf.Variable(dtype=tf.float64, initial_value=float("inf"), trainable=False,
                                                         name="training_epoch_perplexity")
            tf.summary.scalar(self.training_epoch_perplexity.op.name, self.training_epoch_perplexity)
            

            self.validation_accuracy = tf.Variable(dtype=tf.float64, initial_value=float("0"), trainable=False,
                                                     name="validation_accuracy")
            tf.summary.scalar(self.validation_accuracy.op.name, self.validation_accuracy)
            self.training_epoch_accuracy = tf.Variable(dtype=tf.float64, initial_value=float("0"), trainable=False,
                                                         name="training_epoch_accuracy")
            tf.summary.scalar(self.training_epoch_accuracy.op.name, self.training_epoch_accuracy)
            
            self.validation_recall = tf.Variable(dtype=tf.float64, initial_value=float("0"), trainable=False,
                                                     name="validation_recall")
            tf.summary.scalar(self.validation_recall.op.name, self.validation_recall)
            self.training_epoch_recall = tf.Variable(dtype=tf.float64, initial_value=float("0"), trainable=False,
                                                         name="training_epoch_recall")
            tf.summary.scalar(self.training_epoch_recall.op.name, self.training_epoch_recall)
            
            
            self.validation_precision = tf.Variable(dtype=tf.float64, initial_value=float("0"), trainable=False,
                                                     name="validation_precision")
            tf.summary.scalar(self.validation_precision.op.name, self.validation_precision)
            self.training_epoch_precision = tf.Variable(dtype=tf.float64, initial_value=float("0"), trainable=False,
                                                         name="training_epoch_precision")
            tf.summary.scalar(self.training_epoch_precision.op.name, self.training_epoch_precision)
            
            
            self.validation_f1_score = tf.Variable(dtype=tf.float64, initial_value=float("0"), trainable=False,
                                                     name="validation_f1_score")
            tf.summary.scalar(self.validation_f1_score.op.name, self.validation_f1_score)
            self.training_epoch_f1_score = tf.Variable(dtype=tf.float64, initial_value=float("0"), trainable=False,
                                                         name="training_epoch_f1_score")
            tf.summary.scalar(self.training_epoch_f1_score.op.name, self.training_epoch_f1_score)
            
            
            
            self.iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
            self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()),
                                                       max_gradient, name="clip_gradients")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_step = optimizer.apply_gradients(zip(self.gradients, tf.trainable_variables()),
                                                        name="train_step",
                                                        global_step=self.iteration)
            #self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.summary = tf.summary.merge_all()

    @property
    def batch_size(self):
        return self.input.get_shape()[0].value

    @property
    def time_steps(self):
        return self.input.get_shape()[1].value

    @property
    def vocabulary_size(self):
        return self.embedding.get_shape()[0].value

    @property
    def hidden_units(self):
        return self.embedding.get_shape()[1].value

    @property
    def output_size(self):
        return self.labels.get_shape()[2].value
    
    def _set_state(self, feed_dict, state):
        if isinstance(self.state,list):
            for i in range(len(self.state)):
                feed_dict[self.state[i]]=state[i]
        else:
            feed_dict[self.state] = state
        return feed_dict
    
    def getVocabulary(self, X):
        vocabulary = {'padding':0, 'UKN': 1}
        instr_index = 2
        for X_seq in X:
            for instruction in X_seq:
                if not instruction in vocabulary:
                    vocabulary[instruction] = instr_index
                    instr_index += 1
        
        
        self.vocabulary = vocabulary
        return vocabulary
    
    def inference(self, session, x,y):
        output_y = np.array([])
        state = None
        for i in range(0, len(x), self.time_steps):
            batch_X = np.zeros((self.batch_size, self.time_steps))
            batch_labels = np.zeros((self.batch_size, self.time_steps, self.output_size))
            batch_mask = np.zeros((self.batch_size, self.time_steps, self.output_size))
            batch_seq_len = np.zeros((self.batch_size,))
            
            seq_len = len(x[i:i+self.time_steps])
            y_seq = y[i:i+self.time_steps]
            batch_X[0,0:seq_len] = x[i:i+self.time_steps]
            batch_labels[0,0:seq_len] = np.eye(self.output_size)[y_seq]
            batch_mask[0,0:seq_len] = np.ones((seq_len, self.output_size))
            batch_seq_len[0] = seq_len

            feed_dict={
                        self.input: batch_X,
                        self.labels: batch_labels,
                        self.seq_len: batch_seq_len,
                        self.mask: batch_mask,
                        self.keep_probability: 1
                    }
            
            if state:
                self._set_state(feed_dict, state)

               
            cost, state, acc, prediced = session.run(
            [self.cost, self.next_state, self.accuracy, self.predicted],
                        feed_dict=feed_dict)
            output_y = np.append(output_y, np.argmax(prediced,axis=1)[0:seq_len])
        
        return output_y.astype(np.int32)

    def train(self, session, training_set, parameters, exit_criteria, validation, logging_interval, directories):
        epoch = 1
        iteration = 0
        state = None
        summary = self.summary_writer(directories.summary, session)
        ret_validation_perplexity, ret_acc = None, None

        self.vocabulary = training_set.vocabulary

        iop = tf.global_variables_initializer()
        loc = tf.local_variables_initializer()
        session.run(iop)
        session.run(loc)

        train_acc = Accuracy()
        valid_acc = Accuracy()

        try:
            # Enumerate over the training set until exit criteria are met.
            while True:
                epoch_cost = epoch_iteration = accuracy = num_units = 0
                epoch_labels = np.array([])
                epoch_pred = np.array([])
                # Enumerate over a single epoch of the training set.
                for start_document, context, labels, seq_len, mask, complete, instruction_values in training_set.epoch(
                                                                                                    self.time_steps, self.batch_size,
                                                                                                    args_dim=self.additional_parameters.args_dim,
                                                                                                    encode_int=self.additional_parameters.encode_int):
                    feed_dict={
                        self.input: context,
                        self.labels: labels,
                        self.seq_len: seq_len,
                        self.mask: mask,
                        self.learning_rate: parameters.learning_rate,
                        self.keep_probability: parameters.keep_probability
                    }
                    if start_document:
                        state = session.run(self.state)
                    else:
                        self._set_state(feed_dict, parameters.state)
                    if self.additional_parameters.args_dim:
                        feed_dict[self.input] = context[0]
                        feed_dict[self.input_args] = context[1]
                    if self.additional_parameters.encode_int:
                        feed_dict[self.input] = context[0]
                        feed_dict[self.encoded_int_args] = context[1]
                    
                    _, cost, state, iteration, acc, predicted = session.run(
                    [self.train_step, self.cost, self.next_state, self.iteration, self.accuracy, self.predicted],
                        feed_dict=feed_dict)

                    #iteration = self.iteration                
                    epoch_cost += cost
                    epoch_iteration += self.batch_size
                    accuracy += acc*np.sum(seq_len)
                    num_units += np.sum(seq_len)
                    predict = np.reshape(predicted,(self.batch_size, self.time_steps, self.output_size))
                    for label_seq, pred_seq, sl in zip(labels, predict, seq_len):
                        sl = int(sl)
                        epoch_labels = np.append(epoch_labels, np.argmax(label_seq[:sl],axis=1))
                        epoch_pred = np.append(epoch_pred, np.argmax(pred_seq[:sl], axis=1))
                    train_acc.evaluate(logits=predict,
                    labels=labels, seq_len=seq_len)
                    if self._interval(iteration, logging_interval):
                        logger.info("Epoch %d (%0.4f complete), Iteration %d: epoch training perplexity %0.4f, acc: %0.4f" %
                                    (epoch, complete, iteration, self.perplexity(epoch_cost, epoch_iteration), accuracy/num_units))
                        num_units = 0
                        accuracy = 0
                        train_acc.save(iteration)
                        
                        '''
                        index = np.argmax(np.sum(labels[:,:,1], axis=(1,)))
                        print_lab = np.argmax(labels[index],axis=1)[:int(seq_len[index])]
                        print_pred = np.argmax(prediced[index*self.time_steps:(index+1)*self.time_steps], axis=1)[:int(seq_len[index])]
                        logger.info(str(instruction_values[index]))
                        logger.info(str(print_lab))
                        logger.info(str(print_pred))
                        '''
                    if validation is not None and self._interval(iteration, validation.interval):
                        validation_perplexity, acc, val_lab, val_pred = self.test(session, validation.validation_set, valid_acc)
                        self.store_validation_perplexity(session, summary, iteration, validation_perplexity)
                        self.store_validation_accuracy(session, summary, iteration, self.accuracy_score(val_lab, val_pred))
                        self.store_validation_recall(session, summary, iteration, self.recall(val_lab, val_pred))
                        self.store_validation_precision(session, summary, iteration, self.precision(val_lab, val_pred))
                        self.store_validation_f1_score(session, summary, iteration, self.f1_score(val_lab, val_pred))

                        logger.info("Epoch %d, Iteration %d: validation perplexity %0.4f, acc: %0.4f" %
                                    (epoch, iteration, validation_perplexity, acc))
                        valid_acc.save(iteration)
                        self.acc = max(self.acc, acc)
                        #index = np.argmax([np.sum(y) for y in validation.validation_set.Y])
                        #y = np.array(validation.validation_set.Y[index])[:,1]
                        #x = validation.validation_set.encode(validation.validation_set.X[index])
                        #pred = self.inference(session, x, y)
                        #print(validation.validation_set.X[index])
                        #print(str(y))
                        #print(str(pred))
                    if exit_criteria.max_iterations is not None and iteration > exit_criteria.max_iterations:
                        ret_validation_perplexity, ret_acc = self.test(session, validation.validation_set, train=True)
                        logger.info("Epoch %d, Iteration %d: validation perplexity %0.4f, acc: %0.4f" %
                                    (epoch, iteration, ret_validation_perplexity, ret_acc))
                        raise StopTrainingException()

                self.store_training_epoch_perplexity(session, summary, iteration, self.perplexity(epoch_cost, epoch_iteration))
                self.store_training_epoch_accuracy(session, summary, iteration, self.accuracy_score(epoch_labels, epoch_pred))
                self.store_training_epoch_recall(session, summary, iteration, self.recall(epoch_labels, epoch_pred))
                self.store_training_epoch_precision(session, summary, iteration, self.precision(epoch_labels, epoch_pred))
                self.store_training_epoch_precision(session, summary, iteration, self.f1_score(epoch_labels, epoch_pred))
                
                epoch += 1
                if exit_criteria.max_epochs is not None and epoch > exit_criteria.max_epochs:
                    raise StopTrainingException()
        except (StopTrainingException, KeyboardInterrupt):
            pass
        logger.info("Stop training at epoch %d, iteration %d" % (epoch, iteration))
        summary.close()
        if directories.model is not None:
            model_filename = self._model_file(directories.model)
            tf.train.Saver().save(session, model_filename)
            self._write_model_parameters(directories.model)
            self._write_vocabulary(directories.model)
            logger.info("Saved model in %s " % directories.model)
        
        return ret_validation_perplexity, ret_acc

    def _write_model_parameters(self, model_directory):
        parameters = {
            "max_gradient": self.max_gradient,
            "batch_size": self.batch_size,
            "time_steps": self.time_steps,
            "vocabulary_size": self.vocabulary_size,
            "hidden_units": self.hidden_units,
            "output_size": self.output_size,
            "layers": self.layers,
            "args_dim": self.additional_parameters.args_dim,
            "bidirectional": self.additional_parameters.bidirectional,
            "nonlinear": self.additional_parameters.nonlinear,
            "encode_int": self.additional_parameters.encode_int,
            "best_valid_acc": self.acc
        }
        with open(self._parameters_file(model_directory), "w") as f:
            json.dump(parameters, f, indent=4)

    def _write_vocabulary(self, model_directory):
        with open(self._vocabulary_file(model_directory), "w") as f:
            json.dump(self.vocabulary, f, indent=4)
        
        
    def test(self, session, test_set, acc_obj=Accuracy(), train = False):
        state = None       
        epoch_cost_val = epoch_iteration_val = accuracy = num_units = 0
        valid_labels = np.array([])
        valid_pred = np.array([])
        for start_document, context, labels, seq_len, mask, _, instruction_values in test_set.epoch(
                                                                                    self.time_steps, self.batch_size, self.vocabulary,
                                                                                    args_dim=self.additional_parameters.args_dim,
                                                                                    encode_int=self.additional_parameters.encode_int):
            feed_dict={
                self.input: context,
                self.labels: labels,
                self.seq_len: seq_len,
                self.mask: mask,
                self.keep_probability: 1
            }
            if start_document:
                state = session.run(self.state)
            else:
                self._set_state(feed_dict, parameters.state)
            if self.additional_parameters.args_dim:
                feed_dict[self.input] = context[0]
                feed_dict[self.input_args] = context[1]
            if self.additional_parameters.encode_int:
                feed_dict[self.input] = context[0]
                feed_dict[self.encoded_int_args] = context[1]

            if train:
                _ , _, cost, state, acc, pred = session.run([self.train_step, self.iteration, self.cost, self.next_state, self.accuracy, self.predicted],
                                            feed_dict=feed_dict)
            else:
                _, cost, state, acc, pred = session.run([self.iteration, self.cost, self.next_state, self.accuracy, self.predicted],
                                            feed_dict=feed_dict)
            
            epoch_cost_val += cost
            accuracy += acc*np.sum(seq_len)
            num_units += np.sum(seq_len)
            epoch_iteration_val += self.batch_size
            predict = np.reshape(pred,(self.batch_size, self.time_steps, self.output_size))
            acc_obj.evaluate(logits=predict,
                    labels=labels, seq_len=seq_len)
            
            for label_seq, pred_seq, sl in zip(labels, predict, seq_len):
                sl = int(sl)
                valid_labels = np.append(valid_labels, np.argmax(label_seq[:sl],axis=1))
                valid_pred = np.append(valid_pred, np.argmax(pred_seq[:sl], axis=1))

        return self.perplexity(epoch_cost_val, epoch_iteration_val), accuracy/num_units, valid_labels, valid_pred

    @staticmethod
    def _interval(iteration, interval):
        return interval is not None and iteration > 1 and iteration % interval == 0

    @staticmethod
    def perplexity(cost, iterations):
        return np.exp(cost / iterations)
    
    @staticmethod
    def accuracy_score(labels, predictions):
        return accuracy_score(labels, predictions)
    
    @staticmethod
    def recall(labels, predictions):
        return recall_score(labels, predictions, average='macro')
    
    @staticmethod
    def precision(labels, predictions):
        return precision_score(labels, predictions, average='macro')

    @staticmethod
    def f1_score(labels, predictions):
        return f1_score(labels, predictions, average='macro')
        

    def store_validation_perplexity(self, session, summary, iteration, validation_perplexity):
        session.run(self.validation_perplexity.assign(validation_perplexity))
        summary.add_summary(session.run(self.summary), global_step=iteration)

    def store_validation_accuracy(self, session, summary, iteration, accuracy):
        session.run(self.validation_accuracy.assign(accuracy))
        summary.add_summary(session.run(self.summary), global_step=iteration)

    def store_validation_recall(self, session, summary, iteration, recall):
        session.run(self.validation_recall.assign(recall))
        summary.add_summary(session.run(self.summary), global_step=iteration)
    
    def store_validation_precision(self, session, summary, iteration, precision):
        session.run(self.validation_precision.assign(precision))
        summary.add_summary(session.run(self.summary), global_step=iteration)
    
    def store_validation_f1_score(self, session, summary, iteration, f1_score):
        session.run(self.validation_f1_score.assign(f1_score))
        summary.add_summary(session.run(self.summary), global_step=iteration)

    def store_training_epoch_perplexity(self, session, summary, iteration, training_perplexity):
        session.run(self.training_epoch_perplexity.assign(training_perplexity))
        summary.add_summary(session.run(self.summary), global_step=iteration)

    def store_training_epoch_accuracy(self, session, summary, iteration, accuracy):
        session.run(self.training_epoch_accuracy.assign(accuracy))
        summary.add_summary(session.run(self.summary), global_step=iteration)

    def store_training_epoch_recall(self, session, summary, iteration, recall):
        session.run(self.training_epoch_recall.assign(recall))
        summary.add_summary(session.run(self.summary), global_step=iteration)
    
    def store_training_epoch_precision(self, session, summary, iteration, precision):
        session.run(self.training_epoch_precision.assign(precision))
        summary.add_summary(session.run(self.summary), global_step=iteration)
    
    def store_training_epoch_f1_score(self, session, summary, iteration, f1_score):
        session.run(self.training_epoch_f1_score.assign(f1_score))
        summary.add_summary(session.run(self.summary), global_step=iteration)


    @staticmethod
    def summary_writer(summary_directory, session):
        class NullSummaryWriter(object):
            def add_summary(self, *args, **kwargs):
                pass

            def flush(self):
                pass

            def close(self):
                pass

        if summary_directory is not None:
            return tf.summary.FileWriter(summary_directory, session.graph)
        else:
            return NullSummaryWriter()
        


class StopTrainingException(Exception):
    pass


# Objects used to group training parameters
class ExitCriteria(object):
    def __init__(self, max_iterations, max_epochs):
        self.max_iterations = max_iterations
        self.max_epochs = max_epochs


class Parameters(object):
    def __init__(self, learning_rate, keep_probability):
        self.learning_rate = learning_rate
        self.keep_probability = keep_probability


class Validation(object):
    def __init__(self, interval, validation_set):
        self.interval = interval
        self.validation_set = validation_set


class Directories(object):
    def __init__(self, model, summary):
        self.model = model
        self.summary = summary
