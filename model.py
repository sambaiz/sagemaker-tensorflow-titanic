import tensorflow as tf
import logging

class Model:
    def __init__(self, logger: logging.Logger, dropout: float):
        self.logger = logger
        self.model = tf.keras.Sequential([
            tf.keras.layers.DenseFeatures(self._feature_columns()),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def _feature_columns(self):
        return [
            tf.feature_column.numeric_column('age'),
            tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity('sex', 2)),
            tf.feature_column.numeric_column('fare')
        ]

    def _fill(self, feature: str, value):
        def __fill(x):
            if x[feature] == -1.0:
                x[feature] = value
            return x

        return __fill

    def _preprocess(self, ds: tf.data.Dataset):
        ds_size = len(list(ds.filter(lambda x: x['age'] != -1.0)))
        avg_age = ds.filter(lambda x: x['age'] != -1.0).reduce(0.0, lambda x, y: x + y['age']) / ds_size
        ds = ds.map(self._fill('age', avg_age))
        return ds.shuffle(1000).batch(100).prefetch(5)

    def train(self, data_path: str):
        ds = tf.data.experimental.make_csv_dataset(data_path, num_epochs=1, batch_size=1).unbatch()
        for features in self._preprocess(ds):
            self.model.fit(features, tf.one_hot(features['survived'], 2))
            loss, accuracy = self.model.evaluate(features, tf.one_hot(features['survived'], 2), verbose=0)
            self.logger.info(f'train accuracy: {accuracy}, train loss: {loss};')

    def save(self, path: str):
        self.model.save(path, save_format="tf")
