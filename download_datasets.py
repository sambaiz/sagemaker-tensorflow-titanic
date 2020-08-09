import tensorflow_datasets as tfds

if __name__ == '__main__':
    ds_train = tfds.load('titanic', split='train', data_dir='./dataset')