import tensorflow_datasets as tfds
import csv
import os

if __name__ == '__main__':
    ds = tfds.load('titanic', split='train')
    os.makedirs('dataset', exist_ok=True)
    with open('dataset/titanic.csv', 'w') as f:
        writer = csv.writer(f)
        keys = list(ds.element_spec['features'].keys())
        keys.append('survived')
        writer.writerow(keys)
        for feature in ds.as_numpy_iterator():
            features = list(feature['features'].values())
            features.append(feature['survived'])
            writer.writerow(features)
