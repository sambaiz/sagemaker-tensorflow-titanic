import argparse
import os
from model import Model
import json
import logging
import sys


def _make_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def _parse_args():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--dropout', type=float, default=0.2, metavar='DROP', help='dropout rate (default: 0.2)')

    # Container environment
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model'))
    parser.add_argument('--hosts', type=str, default=os.environ.get('SM_HOSTS', '[]'))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    # fit() inputs (SM_CHANNEL_XXXX)
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', 'dataset'))
    args = parser.parse_args()
    args.hosts = json.loads(args.hosts)
    return args


if __name__ == "__main__":
    args = _parse_args()
    logger = _make_logger()
    m = Model(logger, args.dropout)
    m.train(os.path.join(args.data_dir, 'titanic.csv'))
    if len(args.hosts) == 0 or args.current_host == args.hosts[0]:
        m.save(args.model_dir)
