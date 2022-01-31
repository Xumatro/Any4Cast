import json
import math
import numpy as np


class Dataset:
    def __init__(self, settings):
        self.f_width = settings['n_features']
        self.s_width = settings['n_skipped']
        self.l_width = settings['n_labels']
        self.t_width = self.f_width + self.s_width + self.l_width
        self.t_t_split = settings['train_test_split']

    def load_json(self, json_data, extn_func):
        serial_data = [extn_func(entry) for entry in json.loads(json_data)]

        self.load_serial(serial_data)

    def load_serial(self, serial_data):
        self.serial_data = np.array(serial_data, dtype=np.float32)

        self.mu = self.serial_data.mean()
        self.sigma = self.serial_data.std()
        self.min = self.serial_data.min()
        self.max = self.serial_data.max()

    def normalize(self, method):
        if method == "min-max":
            def norm_func(entry):
                return ((entry - self.min) * 2) / (self.max - self.min) - 1
        elif method == "z-score":
            def norm_func(entry):
                return ((entry - self.mu) / self.sigma)
        else:
            raise SystemExit(f"Error: \"{method}\" \
is not an implemented normalization method!")

        for (i, entry) in enumerate(self.serial_data):
            self.serial_data[i] = norm_func(entry)

    def create_datasets(self):
        self.create_windows()

        split_point = int(math.floor(len(self.features) * self.t_t_split))

        x_train = self.features[:split_point]
        y_train = self.labels[:split_point]
        x_test = self.features[split_point:]
        y_test = self.labels[split_point:]

        return x_train, y_train, x_test, y_test

    def create_windows(self):
        features = []
        labels = []

        for i in range(len(self.serial_data) - (self.t_width - 1)):
            features.append([[feature] for feature in
                             self.serial_data[i:(i + self.f_width)]])
            labels.append(self.serial_data[(i + (self.t_width - self.l_width)):
                                           (i + self.t_width)])

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        p = np.random.permutation(len(features))
        self.features = features[p]
        self.labels = labels[p]
