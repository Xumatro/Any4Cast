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

    def normalize(self, method="z-score"):
        self.norm_method = method
        if method == "min-max":
            def norm_func(entry):
                return (entry - self.min) / (self.max - self.min)
        elif method == "z-score":
            def norm_func(entry):
                return ((entry - self.mu) / self.sigma)
        else:
            del self.norm_method
            raise SystemExit(f"Error: \"{method}\" \
is not an implemented normalization method!")

        for (i, entry) in enumerate(self.serial_data):
            self.serial_data[i] = norm_func(entry)

    def denormalize(self, predictions, method="z-score"):
        if self.norm_method == "min-max":
            def denorm_func(entry):
                return (entry * (self.max - self.min)) + self.min
        else:
            def denorm_func(entry):
                return (entry * self.sigma) + self.mu

        return [denorm_func(prediction) for prediction in predictions]

    def create_datasets(self, prediction_mode=False):
        self.create_windows(prediction_mode)
        if prediction_mode:
            split_point = len(self.features)
        else:
            split_point = int(math.floor(len(self.features) * self.t_t_split))

        x_train = self.features[:split_point]
        y_train = self.labels[:split_point]
        x_test = self.features[split_point:]
        y_test = self.labels[split_point:]

        if prediction_mode:
            return np.array([x_train[-1]])
        else:
            return x_train, y_train, x_test, y_test

    def create_windows(self, prediction_mode):
        features = []
        labels = []

        if prediction_mode:
            s_width = 0
            l_width = 0
            t_width = self.f_width
        else:
            s_width = self.s_width
            l_width = self.l_width
            t_width = self.f_width + s_width + l_width

        for i in range(len(self.serial_data) - (t_width - 1)):
            features.append([[feature] for feature in
                             self.serial_data[i:(i + self.f_width)]])
            labels.append(self.serial_data[(i + (t_width - l_width)):
                                           (i + t_width)])

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        if prediction_mode:
            self.features = features
            self.labels = labels
        else:
            p = np.random.permutation(len(features))
            self.features = features[p]
            self.labels = labels[p]
