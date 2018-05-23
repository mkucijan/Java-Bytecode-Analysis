import argparse
import json
import logging
import os

from RnnModel import __version__, logger
from RnnModel.command import train_model, test_model, get_data_set_info
from RnnModel.data import Data, DIR_PATH


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


def create_argument_parser():
    parser = argparse.ArgumentParser(description="RnnModel version %s" % __version__, fromfile_prefix_chars='@')
    parser.add_argument('--version', action='version', version="%(prog)s " + __version__)
    parser.add_argument("--log", default="INFO", help="logging level")
    parser.set_defaults(func=usage(parser))

    subparsers = parser.add_subparsers(title="TensorFlow RNN Language Model")

    train = subparsers.add_parser("train", description="Train an RNN language model.", help="train a language model")
    train.add_argument("data_set", type=Data.deserialize, help="data set")
    train.add_argument("--training_partition", type=real_zero_to_one, default=0.9, help="partition containing training data")
    train.add_argument("--validation-partition", type=real_zero_to_one, default=0.1, help="partition containing validation data")
    train.add_argument("--validation-interval", type=positive_integer, default=1000,
                       help="how often to run the validation set")
    train.add_argument("--model-directory", type=new_directory, help="directory to which to write the model")
    train.add_argument("--summary-directory", type=new_directory,
                       help="directory to which to write a training summary")
    train.add_argument("--time-steps", type=positive_integer, default=100, help="training unrolled time steps")
    train.add_argument("--batch-size", type=positive_integer, default=20, help="training size batch")
    train.add_argument("--hidden-units", type=positive_integer, default=500, help="number of hidden units in the RNN")
    train.add_argument("--layers", type=positive_integer, default=1, help="number of RNN layers")
    train.add_argument("--keep-probability", type=real_zero_to_one, default=0.5,
                       help="probability to keep a cell in a dropout layer")
    train.add_argument("--max-gradient", type=positive_real, default=5, help="value to clip gradients to")
    train.add_argument("--max-iterations", type=positive_integer, help="number of training iterations to run")
    train.add_argument("--logging-interval", type=positive_integer, default=100,
                       help="log and write summary after this many iterations")
    train.add_argument("--max-epochs", type=positive_integer, default=30, help="number of training epochs to run")
    train.add_argument("--learning-rate", type=positive_real, default=1.0, help="training learning rate")
    #train.add_argument("--init", type=positive_real, default=0.05, help="random initial absolute value range")
    train.set_defaults(func=train_model)

    test = subparsers.add_parser("test", description="Use an RNN model.",
                                 help="Use a previously-trained model to get perplexity on a test set")
    test.add_argument("model_directory", help="directory from which to read the model")
    test.add_argument("data_set", type=Data.deserialize, default=Data.deserialize(os.path.join(DIR_PATH, 'output')), help="data set")
    test.add_argument("--test-partition",type=real_zero_to_one, default=0.2, help="test partition")
    test.set_defaults(func=test_model)

    sample = subparsers.add_parser("sample", description="Sample text from a RNN model.",
                                   help="sample text from language model")
    sample.add_argument("model", help="directory from which to read the model")
    sample.set_defaults(func=lambda a: print(a))
    return parser


def usage(parser):
    class UsageClosure(object):
        def __call__(self, _):
            parser.print_usage()
            parser.exit(0)

    return UsageClosure()


# noinspection PyShadowingBuiltins
def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)


# Various argparse type functions.

def positive_integer(i):
    i = int(i)
    if i <= 0:
        raise argparse.ArgumentTypeError("%d must be greater than zero" % i)
    return i


def positive_real(r):
    r = float(r)
    if r <= 0:
        raise argparse.ArgumentTypeError("%d must be greater than zero" % r)
    return r


def real_zero_to_one(r):
    r = float(r)
    if r < 0 or r > 1:
        raise argparse.ArgumentTypeError("%f must be between zero and one" % r)
    return r


def new_directory(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        logger.warn("The directory %s already exists. Overwriting!" % directory)
        #raise argparse.ArgumentError(None, "The directory %s already exists." % directory)
    return directory


def json_file(filename):
    with open(filename) as f:
        return json.load(f)


if __name__ == '__main__':
    main()
