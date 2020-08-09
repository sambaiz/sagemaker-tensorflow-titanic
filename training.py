import argparse
import os
from model import Model
import tensorflow_datasets as tfds


def _parse_args():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--dropout', type=float, default=0.2, metavar='DROP', help='dropout rate (default: 0.2)')

    # Container environment
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))
    parser.add_argument('--hosts', type=str, default=os.environ.get('SM_HOSTS', '[]'))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    # fit() inputs (SM_CHANNEL_XXXX)
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', 'dataset'))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    m = Model(args.dropout)
    ds = tfds.load('titanic', split='train', data_dir=args.data_dir)
    m.train(ds)
    if args.current_host == args.hosts[0]:
        m.save(os.path.join(args.sm_model_dir, 'model.h5'))
