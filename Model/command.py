import time
from datetime import timedelta

import tensorflow as tf
from RnnModel import logger
from RnnModel.rnn import RNN, Parameters, ExitCriteria, Validation, Directories
from RnnModel.data import Data


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

    args.data_set.relabelData(overwrite=True)
    logger.info("Baseline acc: %0.4f" % args.data_set.baseline)
    
    train_data = args.data_set.getPartition(args.training_partition)
    if args.validation_partition:
        validation = Validation(args.validation_interval, args.data_set.getPartition(args.validation_partition))
    else:
        validation = None
    # Run training.
    start_time = time.time()
    with tf.Graph().as_default():
        model = RNN(args.max_gradient,
                    args.batch_size, args.time_steps, train_data.vocabulary_size,
                    args.hidden_units,args.data_set.output_size, args.layers)

        with tf.device('device:GPU:0'):
            with tf.Session() as session:
                model.train(session,
                            train_data,
                            Parameters(args.learning_rate, args.keep_probability),
                            ExitCriteria(args.max_iterations, args.max_epochs),
                            validation,
                            args.logging_interval,
                            Directories(args.model_directory, args.summary_directory))
    logger.info("Total training time %s" % timedelta(seconds=(time.time() - start_time)))


def test_model(args):
    test_set = args.data_set[args.test_partition]
    logger.info("Test set: %s" % test_set)
    with tf.Graph().as_default():
        with tf.Session() as session:
            model = RNN.restore(session, args.model_directory)
            perplexity = model.test(session, test_set)
            print("Perplexity %0.4f" % perplexity)
