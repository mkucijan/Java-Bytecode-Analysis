import time
from datetime import timedelta
import itertools
import numpy as np

import tensorflow as tf
from Model import logger
from Model.rnn import RNN, Additional_Parameters, Parameters, ExitCriteria, Validation, Directories
from Model.data import Data


def get_data_set_info(args):
    print(args.data_set)
    if args.batches is not None:
        print()
        time_steps = args.batches[0]
        batch_size = args.batches[1]
        for partition_name in args.data_set:
            print("%s %d batches" %
                  (partition_name, args.data_set[partition_name].total_batches(time_steps, batch_size)))


def train_model(args):
    if args.model_directory is None:
        logger.warn("Not saving a model.")
        
    additional_parameters = Additional_Parameters(
        args_dim = args.args_dim,
        bidirectional= args.bidirectional,
        nonlinear= args.nonlinear
    )
    
    args.data_set.shuffle()
    args.data_set.relabelData(overwrite=True)
    
    logger.info("Baseline acc: %0.4f" % args.data_set.baseline)
    train_data = args.data_set.getPartition(args.training_partition)
    logger.info("Train baseline acc: %0.4f" % train_data.baseline)

    if args.validation_partition:
        validation_data = args.data_set.getPartition(args.validation_partition)
        validation = Validation(args.validation_interval, validation_data)
        logger.info("Validation baseline acc: %0.4f" % validation_data.baseline)
    else:
        validation = None
    # Run training.
    
    start_time = time.time()
    with tf.Graph().as_default():
        model = RNN(args.max_gradient,
                    args.batch_size, args.time_steps, train_data.vocabulary_size,
                    args.hidden_units,args.data_set.output_size, args.layers, additional_parameters)

        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.device('/gpu:0'):
            with tf.Session(config = config) as session:
                model.train(session,
                            train_data,
                            Parameters(args.learning_rate, args.keep_probability),
                            ExitCriteria(args.max_iterations, args.max_epochs),
                            validation,
                            args.logging_interval,
                            Directories(args.model_directory, args.summary_directory))
    logger.info("Total training time %s" % timedelta(seconds=(time.time() - start_time)))

def search_params(args):
    if args.model_directory is None:
        logger.warn("Not saving a model.")
        
    additional_parameters = Additional_Parameters(
        args_dim = args.args_dim,
        bidirectional= args.bidirectional,
        nonlinear= args.nonlinear
    )
    
    # check this settings
    args.data_set.shuffle()
    args.data_set.relabelData(overwrite=True)
    ###

    logger.info("Baseline acc: %0.4f" % args.data_set.baseline)
    train_data = args.data_set.getPartition(args.training_partition)
    logger.info("Train baseline acc: %0.4f" % train_data.baseline)

    if args.validation_partition:
        validation_data = args.data_set.getPartition(args.validation_partition)
        validation = Validation(args.validation_interval, validation_data)
        logger.info("Validation baseline acc: %0.4f" % validation_data.baseline)
    else:
        validation = None
    
    
    ms = [0.1, 0.01]
    bs = [16, 32]
    ts = [20, 50 ,100]
    hu = [128, 256, 512, 650]
    ly = [1, 2, 3, 4]
    ad = [None, 16, 32, 64]
    bi = [True, False]
    nl = [True, False]

    perplexity_values = []
    accuracy_values = []

    for params in itertools.product(*[ms, bs, ts, hu, ly, ad, bi, nl]):
        args.max_gradient,                    \
        args.batch_size,                      \
        args.time_steps,                      \
        args.hidden_units,                    \
        args.layers,                          \
        additional_parameters.args_dim,       \
        additional_parameters.bidirectional,  \
        additional_parameters.nonlinear       \
        = params
    
        start_time = time.time()
        with tf.Graph().as_default():
            model = RNN(args.max_gradient,
                        args.batch_size, args.time_steps, train_data.vocabulary_size,
                            args.hidden_units,args.data_set.output_size, args.layers,
                            additional_parameters)

            config = tf.ConfigProto(allow_soft_placement = True)
            with tf.device('/gpu:0'):
                with tf.Session(config = config) as session:
                    val, acc = model.train(session,
                                           train_data,
                                           Parameters(args.learning_rate, args.keep_probability),
                                           ExitCriteria(args.max_iterations, args.max_epochs),
                                           validation,
                                           args.logging_interval,
                                           Directories(args.model_directory, args.summary_directory))
        logger.info("Total training time %s" % timedelta(seconds=(time.time() - start_time)))
    
        perplexity_values.append(val)
        accuracy_values.append(acc)
    
    logger.info("Perplexity: " + str(perplexity_values))
    logger.info("Accuracy: " + str(accuracy_values))
    best_iteration = np.argmax(accuracy_values)
    best_params = list(itertools.product(*[ms, bs, ts, hu, ly, ad, bi, nl]))[best_iteration]
    logger.warn("Best Iteration: "+str(best_iteration))
    param_names = [
    'max_gradient',
    'batch_size',              
    'time_steps',                 
    'hidden_units',
    'layer',
    'args_dim',  
    'bidirectional',
    'nonlinear'   ]
    result = ''
    for name, param in zip(param_names, best_params):
        result += '\t\t\t'+name+': '+str(param)+'\n'
    logger.warn("Best Params:\n"+result)




def test_model(args):
    test_set = args.data_set[args.test_partition]
    logger.info("Test set: %s" % test_set)
    with tf.Graph().as_default():
        with tf.Session() as session:
            model = RNN.restore(session, args.model_directory)
            perplexity = model.test(session, test_set)
            print("Perplexity %0.4f" % perplexity)