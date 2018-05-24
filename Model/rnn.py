import json
import os.path

import numpy as np
import tensorflow as tf
from RnnModel import logger


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
        model = cls(parameters["max_gradient"],
                    parameters["batch_size"], parameters["time_steps"], parameters["vocabulary_size"],
                    parameters["hidden_units"], parameters["output_size"], parameters["layers"])
        tf.train.Saver().restore(session, cls._model_file(model_directory))
        return model

    @staticmethod
    def _parameters_file(model_directory):
        return os.path.join(model_directory, "parameters.json")

    @staticmethod
    def _model_file(model_directory):
        return os.path.join(model_directory, "model")

    def __init__(self, 
                 max_gradient,
                 batch_size,
                 time_steps,
                 vocabulary_size,
                 hidden_units,
                 output_size,
                 layers,
                 bidirectional=True,
                 nonlinear=True):
        
        self.max_gradient = max_gradient
        self.layers = layers
        
        # Add vocabulary slots of out of vocabulary (index 1) and padding (index 0).
        # vocabulary_size += 2

        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        with tf.name_scope("Parameters"):
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")

        with tf.name_scope("Input"):
            self.input = tf.placeholder(tf.int32, shape=(batch_size, time_steps), name="input")
            self.labels = tf.placeholder(tf.float32, shape=(batch_size, time_steps, output_size), name="targets")
            self.mask = tf.placeholder(tf.float32, shape=(batch_size, time_steps, output_size), name="mask")
            self.seq_len = tf.placeholder(tf.int32, [batch_size,])
        
        with tf.name_scope("Embedding"):
            self.embedding = tf.get_variable("embedding", (vocabulary_size, hidden_units), 
                                            initializer = tf.contrib.layers.xavier_initializer(),
                                            dtype=tf.float32) 
            #self.embedding = initializer((vocabulary_size, hidden_units))
            self.embedded_input = tf.nn.embedding_lookup(self.embedding, self.input, name="embedded_input")

        if bidirectional:
            with tf.name_scope("RNN_Bidirectional"):
                cells_fw = [
                    tf.nn.rnn_cell.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(hidden_size,activation=tf.nn.tanh),
                    output_keep_prob = self.keep_probability)
                    for _ in range(layers)]
    
                cells_bw = [
                    tf.nn.rnn_cell.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(hidden_size,activation=tf.nn.tanh),
                    output_keep_prob = self.keep_probability)
                    for _ in range(layers)]
                
                self.state = ([cell.zero_state(batch_size, dtype=tf.float32) for cell in cells_fw],
                              [cell.zero_state(batch_size, dtype=tf.float32) for cell in cells_bw])
                outputs, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                cells_fw = cells_fw,
                                                cells_bw=cells_bw,
                                                initial_states_fw = self.state[0],
                                                initial_states_bw = self.state[1],
                                                sequence_length = self.seq_len,
                                                dtype = tf.float32,
                                                inputs = self.embedded_input)
                
                self.next_state = (states_fw, states_bw)
                self.outputs_rnn = tf.concat(outputs, 2)
                hidden_layer_size = hidden_units*2
        else:
            with tf.name_scope("RNN"):
                cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_probability)
                rnn_layers = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)
                self.state = rnn_layers.zero_state(batch_size, dtype=tf.float32)
                self.outputs_rnn, self.next_state = tf.nn.dynamic_rnn(rnn_layers, self.embedded_input,
                                                                sequence_length=self.seq_len,
                                                                initial_state=self.state)
                hidden_layer_size = hidden_units
        if nonlinear:
            with tf.name_scope("Nonlinear_output"):
                self.w_nl = tf.get_variable("w_nl", (1, hidden_layer_size, hidden_layer_size),
                                            initializer = tf.contrib.layers.xavier_initializer(),
                                            dtype=tf.float32)
                self.b_nl = tf.get_variable("b_nl", hidden_layer_size, initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                self.outputs = tf.nn.relu(tf.matmul(self.outputs_rnn, tf.tile(self.w_nl, [tf.shape(self.outputs_rnn)[0],1,1])) + self.b_nl)
        else:
            self.outputs = self.outputs_rnn
        with tf.name_scope("Cost"):
            #self.w = tf.Variable(initializer((hidden_size, output_size)), dtype = tf.float32, name = "W")
            #self.b = tf.Variable(initializer((output_size)), dtype = tf.float32, name = "b")
            # Concatenate all the batches into a single row.
            self.flattened_outputs = tf.reshape(self.outputs, [-1, tf.shape(self.outputs)[2]],
                                                name="flattened_outputs")
            self.flattened_labels = tf.reshape(self.labels, [-1, output_size],
                                                name="flattened_labels")
            self.flattened_mask = tf.reshape(self.mask, [-1, output_size],
                                                name="flattened_mask")
            # Project the outputs onto the vocabulary.
            self.w = tf.get_variable("w", (hidden_layer_size, output_size), initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.b = tf.get_variable("b", output_size, initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.predicted = tf.matmul(self.flattened_outputs, self.w) + self.b
            self.predicted = tf.multiply(self.predicted, self.flattened_mask)
            # Compare predictions to labels.
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.flattened_labels, logits=self.predicted, name="loss")
            self.cost = tf.div(tf.reduce_sum(self.loss), batch_size, name="cost")
            batch_num = tf.div(tf.reduce_sum(self.mask), output_size) 
            padding_num = batch_size*time_steps - batch_num
            correct_num = tf.reduce_sum( tf.cast(tf.equal(tf.argmax(self.flattened_labels, axis=1), tf.argmax(self.predicted, axis=1)), tf.float32) ) - padding_num
            self.accuracy = correct_num/batch_num
        with tf.name_scope("Train"):
            self.validation_perplexity = tf.Variable(dtype=tf.float32, initial_value=float("inf"), trainable=False,
                                                     name="validation_perplexity")
            tf.summary.scalar(self.validation_perplexity.op.name, self.validation_perplexity)
            self.training_epoch_perplexity = tf.Variable(dtype=tf.float32, initial_value=float("inf"), trainable=False,
                                                         name="training_epoch_perplexity")
            tf.summary.scalar(self.training_epoch_perplexity.op.name, self.training_epoch_perplexity)
            self.iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
            self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()),
                                                       max_gradient, name="clip_gradients")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_step = optimizer.apply_gradients(zip(self.gradients, tf.trainable_variables()),
                                                        name="train_step",
                                                        global_step=self.iteration)

        self.initialize = tf.initialize_all_variables()
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

    def train(self, session, training_set, parameters, exit_criteria, validation, logging_interval, directories):
        epoch = 1
        iteration = 0
        state = None
        summary = self.summary_writer(directories.summary, session)

        iop = tf.global_variables_initializer()
        loc = tf.local_variables_initializer()
        session.run(iop)
        session.run(loc)
        try:
            # Enumerate over the training set until exit criteria are met.
            while True:
                epoch_cost = epoch_iteration = accuracy = num_batches = 0
                # Enumerate over a single epoch of the training set.
                for start_document, context, labels, seq_len, mask, complete in training_set.epoch(self.time_steps, self.batch_size):
                    if start_document:
                        state = session.run(self.state)
                    _, cost, state, iteration, acc = session.run(
                        [self.train_step, self.cost, self.next_state, self.iteration, self.accuracy],
                        feed_dict={
                            self.input: context,
                            self.labels: labels,
                            self.seq_len: seq_len,
                            self.mask: mask,
                            self.state: state,
                            self.learning_rate: parameters.learning_rate,
                            self.keep_probability: parameters.keep_probability
                        })
                    epoch_cost += cost
                    epoch_iteration += sum(seq_len)
                    accuracy += acc
                    num_batches += 1
                    if self._interval(iteration, logging_interval):
                        logger.info("Epoch %d (%0.4f complete), Iteration %d: epoch training perplexity %0.4f, acc: %0.4f" %
                                    (epoch, complete, iteration, self.perplexity(epoch_cost, epoch_iteration), accuracy/num_batches))
                        num_batches = 0
                        accuracy = 0
                    if validation is not None and self._interval(iteration, validation.interval):
                        validation_perplexity, acc = self.test(session, validation.validation_set)
                        self.store_validation_perplexity(session, summary, iteration, validation_perplexity)
                        logger.info("Epoch %d, Iteration %d: validation perplexity %0.4f, acc: %0.4f" %
                                    (epoch, iteration, validation_perplexity, acc))
                    if exit_criteria.max_iterations is not None and iteration > exit_criteria.max_iterations:
                        raise StopTrainingException()

                self.store_training_epoch_perplexity(session, summary, iteration,
                                                     self.perplexity(epoch_cost, epoch_iteration))
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
            logger.info("Saved model in %s " % directories.model)

    def _write_model_parameters(self, model_directory):
        parameters = {
            "max_gradient": self.max_gradient,
            "batch_size": self.batch_size,
            "time_steps": self.time_steps,
            "vocabulary_size": self.vocabulary_size,
            "hidden_units": self.hidden_units,
            "output_size": self.output_size,
            "layers": self.layers
        }
        with open(self._parameters_file(model_directory), "w") as f:
            json.dump(parameters, f, indent=4)

    def test(self, session, test_set):
        state = None
        iop = tf.global_variables_initializer()
        loc = tf.local_variables_initializer()
        session.run(iop)
        session.run(loc)       
        epoch_cost = epoch_iteration = accuracy = num_batches = 0
        for start_document, context, labels, seq_len, mask, _ in test_set.epoch(self.time_steps, self.batch_size):
            if start_document:
                state = session.run(self.state)
            cost, state, acc = session.run([self.cost, self.next_state, self.accuracy],
                                      feed_dict={
                                          self.input: context,
                                          self.labels: labels,
                                          self.seq_len: seq_len,
                                          self.mask: mask,
                                          self.state: state,
                                          self.keep_probability: 1
                                      })
            epoch_cost += cost
            accuracy += acc
            num_batches += 1
            epoch_iteration += self.time_steps
        return self.perplexity(epoch_cost, epoch_iteration), accuracy/num_batches


    @staticmethod
    def _interval(iteration, interval):
        return interval is not None and iteration > 1 and iteration % interval == 0

    @staticmethod
    def perplexity(cost, iterations):
        return np.exp(cost / iterations)

    def store_validation_perplexity(self, session, summary, iteration, validation_perplexity):
        session.run(self.validation_perplexity.assign(validation_perplexity))
        summary.add_summary(session.run(self.summary), global_step=iteration)

    def store_training_epoch_perplexity(self, session, summary, iteration, training_perplexity):
        session.run(self.training_epoch_perplexity.assign(training_perplexity))
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
