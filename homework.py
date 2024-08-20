import csv
from tqdm import tqdm
from copy import deepcopy
import argparse
import os
import numpy as np
import pandas as pd


def load_file(path, kind):
    labels_path = os.path.join(path, '%s_label.csv' % kind)
    data_path = os.path.join(path, '%s_data.csv' % kind)
    data_x = pd.read_csv(data_path)
    final = data_x.drop(['BROKERTITLE', 'ADDRESS', 'STATE', 'MAIN_ADDRESS', 'LOCALITY', 'STREET_NAME', 'LONG_NAME',
                         'FORMATTED_ADDRESS'], axis='columns')
    label_x = pd.read_csv(labels_path)
    return final, label_x

def load_test_file(path, kind):
    data_path = os.path.join(path, '%s_data.csv' % kind)
    data_x = pd.read_csv(data_path)
    final = data_x.drop(['BROKERTITLE', 'ADDRESS', 'STATE', 'MAIN_ADDRESS', 'LOCALITY', 'STREET_NAME', 'LONG_NAME',
                         'FORMATTED_ADDRESS'], axis='columns')
    #label_x = pd.read_csv(labels_path)
    return final

class OneHotEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        self.categories_ = [np.unique(col) for col in X.T]

    def transform(self, X):
        X = np.array(X)
        transformed = []
        for i, col in enumerate(X.T):
            unique_values = self.categories_[i]
            encoding = np.zeros((len(col), len(unique_values)))
            for j, value in enumerate(col):
                # encoding[j, np.where(unique_values == value)] = 1
                indices = np.where(np.isin(unique_values, value))[0]
                encoding[j, indices] = 1
            transformed.append(encoding)
        return np.concatenate(transformed, axis=1)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Scaler:
    def __init__(self):
        self.means_ = None
        self.stds_ = None

    def fit(self, X):
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.means_) / self.stds_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def train_test_split(X_data, Y_data, test_size=0.3, random_state=None):
    num_test = int(test_size * len(X_data))

    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X_data))

    train_indices, test_indices = indices[num_test:], indices[:num_test]

    X_train, X_validation = X_data[train_indices], X_data[test_indices]
    Y_train, Y_validation = Y_data[train_indices], Y_data[test_indices]

    return X_train, X_validation, Y_train, Y_validation


def file_handler(data_dir):
    Xtrain, Ytrain = load_file(data_dir, kind='train')
    Xtest = load_test_file(data_dir, kind='test')
    Xtrain.to_csv('Xtrain.csv', index=False)
    #Xtest.to_csv('Xtest.csv', index=False)
    # num_test = int(test_size * len(X))
    col_to_encode = ['TYPE', 'ADMINISTRATIVE_AREA_LEVEL_2', 'SUBLOCALITY']
    col_to_scale = ['PRICE', 'BATH', 'PROPERTYSQFT', 'LATITUDE', 'LONGITUDE']
    encoder = OneHotEncoder()
    encoder.fit(Xtrain[col_to_encode])
    encoded_train_data = encoder.transform(Xtrain[col_to_encode])
    encoder = OneHotEncoder()
    encoder.fit(Xtest[col_to_encode])
    encoded_test_data = encoder.transform(Xtest[col_to_encode])
    scaler = Scaler()
    scaler.fit(Xtrain[col_to_scale])
    scaled_train_data = scaler.transform(Xtrain[col_to_scale])
    scaler = Scaler()
    scaler.fit(Xtest[col_to_scale])
    scaled_test_data = scaler.transform(Xtest[col_to_scale])
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    Xtest = np.array(Xtest)


    Xtrain = np.concatenate((encoded_train_data, scaled_train_data), axis=1)
    Xtest = np.concatenate((encoded_test_data, scaled_test_data), axis=1)

    Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain)

    return Xtrain, Ytrain, Xvalid, Yvalid, Xtest


def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = []
        for row in reader:
            data.append(row)
    return data


def one_hot_encode(data, col_index):
    unique_values = set(row[col_index] for row in data)
    encoding_dict = {value: idx for idx, value in enumerate(unique_values)}
    num_classes = len(unique_values)
    encoded_data = np.zeros((len(data), num_classes))
    for i, row in enumerate(data):
        encoded_data[i, encoding_dict[row[col_index]]] = 1
    return encoded_data


def fit_transform(train_data, test_data):
    categorical_cols = ['BROKERTITLE', 'TYPE', 'ADDRESS', 'STATE', 'MAIN_ADDRESS',
                        'ADMINISTRATIVE_AREA_LEVEL_2', 'LOCALITY', 'SUBLOCALITY',
                        'STREET_NAME', 'LONG_NAME', 'FORMATTED_ADDRESS']
    encoding_dicts = {}
    for col in categorical_cols:
        unique_values = train_data[col].unique()
        encoding_dict = {value: idx for idx, value in enumerate(unique_values)}
        encoding_dicts[col] = encoding_dict

    for col, encoding_dict in encoding_dicts.items():
        test_data[col] = test_data[col].map(encoding_dict).fillna(-1)

    numerical_cols = ['PRICE', 'BATH', 'PROPERTYSQFT', 'LATITUDE', 'LONGITUDE']
    for col in numerical_cols:
        mean = train_data[col].mean()
        std = train_data[col].std()
        train_data[col] = (train_data[col] - mean) / std
        test_data[col] = (test_data[col] - mean) / std

    return train_data, test_data


def normalize_numerical_features(data):
    numerical_data = np.array(data[:, 3:], dtype=float)
    mean = np.mean(numerical_data, axis=0)
    std = np.std(numerical_data, axis=0)
    normalized_data = (numerical_data - mean) / std
    data[:, 3:] = normalized_data
    return data


class Softmax:
    def __init__(self):
        self.one_hot_y = None
        self.calibrated_logits = None
        self.sum_exp_calibrated_logits = None
        self.probabilities = None

    def forward(self, logits, labels):
        self.one_hot_y = np.zeros(logits.shape).reshape(-1)
        self.one_hot_y[labels.astype(int).reshape(-1) + np.arange(logits.shape[0]) * logits.shape[1]] = 1.0
        self.one_hot_y = self.one_hot_y.reshape(logits.shape)

        self.calibrated_logits = logits - np.amax(logits, axis=1, keepdims=True)
        self.sum_exp_calibrated_logits = np.sum(np.exp(self.calibrated_logits), axis=1, keepdims=True)
        self.probabilities = np.exp(self.calibrated_logits) / self.sum_exp_calibrated_logits

        forward_output = - np.sum(
            np.multiply(self.one_hot_y, self.calibrated_logits - np.log(self.sum_exp_calibrated_logits))) / \
                         logits.shape[0]
        return forward_output

    def backward(self, logits, labels):
        gradient = - (self.one_hot_y - self.probabilities) / logits.shape[0]
        return gradient


def predict_label(f):
    if f.shape[1] == 1:
        return (f > 0).astype(float)
    else:
        return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))


class batch_x_y:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.num_samples, self.num_features = self.features.shape

    def get_example(self, indices):
        batch_features = np.zeros((len(indices), self.num_features))
        batch_labels = np.zeros((len(indices), 1))
        for i in range(len(indices)):
            batch_features[i] = self.features[indices[i]]
            batch_labels[i, :] = self.labels[indices[i]]
        return batch_features, batch_labels

class batch_x_test:
    def __init__(self, features):
        self.features = features
        self.num_samples, self.num_features = self.features.shape

    def get_example(self, indices):
        batch_features = np.zeros((len(indices), self.num_features))
        for i in range(len(indices)):
            batch_features[i] = self.features[indices[i]]
        return batch_features

class A_layer:

    def __init__(self, input_D, output_D):
        self.parameter = dict()
        self.gradient = dict()
        self.parameter['W'] = np.random.normal(0, 0.1, size=(input_D, output_D))
        self.parameter['b'] = np.random.normal(0, 0.1, size=output_D)
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros(output_D)

    def forward(self, X):
        W = self.parameter['W']
        b = self.parameter['b']
        forward_output = X @ W + b
        return forward_output

    def backward(self, X, grad):
        W = self.parameter['W']
        self.gradient['W'] = X.T @ grad
        self.gradient['b'] = np.sum(grad, axis=0)
        backward_output = grad @ W.T
        return backward_output


class relu:

    def __init__(self):
        self.mask = None

    def forward(self, X):
        forward_output = np.maximum(0, X)
        self.mask = X > 0
        return forward_output

    def backward(self, X, grad):
        backward_output = grad * self.mask
        return backward_output


def GD(model, _learning_rate):
    for module_name, module in model.items():
        if hasattr(module, 'parameter'):
            for key, _ in module.parameter.items():
                g = module.gradient[key]

                val = module.parameter[key]
                val = val - _learning_rate * g
                module.parameter[key] = val
    return model


def forward_pass(model, x, y):
    l1 = model['L1'].forward(x)
    nl1 = model['nonlinear1'].forward(l1)
    l2 = model['L2'].forward(nl1)
    loss = model['loss'].forward(l2, y)

    return l1, nl1, l2, loss

def forward_pass_test(model, x):
    l1 = model['L1'].forward(x)
    nl1 = model['nonlinear1'].forward(l1)
    l2 = model['L2'].forward(nl1)


    return l1, nl1, l2

def backward_pass(model, x, l1, nl1, l2, y):
    grad_l2 = model['loss'].backward(l2, y)
    grad_nl1 = model['L2'].backward(nl1, grad_l2)
    grad_l1 = model['nonlinear1'].backward(l1, grad_nl1)
    grad_x = model['L1'].backward(x, grad_l1)


def compute_accuracy_loss(N_data, DataSet, model, minibatch_size=100):
    acc = 0.0
    loss = 0.0
    count = 0

    for i in range(int(np.floor(N_data / minibatch_size))):
        x, y = DataSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))
        # print (y)
        _, _, l2, batch_loss = forward_pass(model, x, y)
        loss += batch_loss
        acc += np.sum(predict_label(l2) == y)
        count += len(y)

    return acc / count, loss


def mgc(DataSet, model):
    x, y = DataSet.get_example(np.arange(5))

    l1, nl1, l2, _ = forward_pass(model, x, y)
    backward_pass(model, x, l1, nl1, l2, y)

    x, y = DataSet.get_example(np.arange(500))

    l1, nl1, l2, _ = forward_pass(model, x, y)
    backward_pass(model, x, l1, nl1, l2, y)


def checkG(DataSet, model):
    x, y = DataSet.get_example([0])

    l1, nl1, l2, _ = forward_pass(model, x, y)
    backward_pass(model, x, l1, nl1, l2, y)

    grad_dict = {}
    grad_dict["L1_W_grad_first_dim"] = model['L1'].gradient["W"][0][0]
    grad_dict["L1_b_grad_first_dim"] = model['L1'].gradient["b"][0]
    grad_dict["L2_W_grad_first_dim"] = model['L2'].gradient["W"][0][0]
    grad_dict["L2_b_grad_first_dim"] = model['L2'].gradient["b"][0]

    for name, grad in grad_dict.items():
        layer_name = name.split("_")[0]
        param_name = name.split("_")[1]

        epsilon_value = 1e-3
        epsilon = np.zeros(model[layer_name].parameter[param_name].shape)
        if len(epsilon.shape) == 2:
            epsilon[0][0] = epsilon_value
        else:
            epsilon[0] = epsilon_value

        model[layer_name].parameter[param_name] += epsilon
        _, _, _, f_w_add_epsilon = forward_pass(model, x, y)

        model[layer_name].parameter[param_name] -= 2 * epsilon
        _, _, _, f_w_subtract_epsilon = forward_pass(model, x, y)
        model[layer_name].parameter[param_name] += epsilon


def predict(dataSet, model, minibatch_size=1100):
    predictions = []
    num_batches = len(dataSet.features) // minibatch_size
    remaining_samples = len(dataSet.features) % minibatch_size

    for i in range(num_batches):
        x = dataSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))
        _, _, l2 = forward_pass_test(model, x)

        pred_labels = predict_label(l2)
        predictions.extend(pred_labels)

    if remaining_samples > 0:
        x = dataSet.get_example(np.arange(num_batches * minibatch_size, len(dataSet.features)))
        _, _, l2 = forward_pass_test(model, x)
        pred_labels = predict_label(l2)
        predictions.extend(pred_labels)

    return predictions


def main(main_params):
    np.random.seed(int(main_params['random_seed']))
    Xtrain, Ytrain, Xval, Yval, Xtest  = file_handler(data_dir=main_params['data_dir'])

    Xtest = np.array(Xtest)
    N_train, d = Xtrain.shape

    N_val, _ = Xval.shape

    N_test, _ = Xtest.shape

    trainSet = batch_x_y(Xtrain, Ytrain)
    valSet = batch_x_y(Xval, Yval)
    testSet = batch_x_test(Xtest)

    model = dict()
    num_L1 = 1000
    num_L2 = 100

    epoch_count = int(main_params['epoch_count'])
    minibatch_size = int(main_params['minibatch_size'])
    check_gradient = main_params['check_gradient']
    check_magnitude = main_params['check_magnitude']
    patience = int(main_params['early_stopping_patience'])

    train_accuracy_record = []
    train_loss_record = []
    validation_accuracy_record = []
    val_loss_record = []
    best_epoch = 0

    best_model = None

    _learning_rate = float(main_params['learning_rate'])
    _step = 10

    model['L1'] = A_layer(input_D=d, output_D=num_L1)
    model['nonlinear1'] = relu()
    model['L2'] = A_layer(input_D=num_L1, output_D=num_L2)
    model['loss'] = Softmax()

    if check_magnitude:
        mgc(trainSet, model)

    if check_gradient:
        checkG(trainSet, model)

    # start_time = time.time()

    for t in range(epoch_count):

        idx_order = np.random.permutation(N_train)

        for i in tqdm(range(int(np.floor(N_train / minibatch_size)))):
            x, y = trainSet.get_example(idx_order[i * minibatch_size: (i + 1) * minibatch_size])

            l1, nl1, l2, _ = forward_pass(model, x, y)

            backward_pass(model, x, l1, nl1, l2, y)

            model = GD(model, _learning_rate)

        train_accuracy, train_loss = compute_accuracy_loss(N_train, trainSet, model)
        train_accuracy_record.append(train_accuracy)
        train_loss_record.append(train_loss)

        validation_accuracy, val_loss = compute_accuracy_loss(N_val, valSet, model)
        validation_accuracy_record.append(validation_accuracy)
        val_loss_record.append(val_loss)

        if validation_accuracy == max(validation_accuracy_record):
            best_model = deepcopy(model)
            best_epoch = t + 1
            patience = int(main_params['early_stopping_patience'])
        else:
            patience -= 1

        if patience == 0:
            break

    test_predictions = predict(testSet, best_model)
    counter = 0

    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["BEDS"])  # Write the header
        for prediction in test_predictions:
            #writer.writerow([int(prediction)])
            writer.writerow([int(prediction.item())])
            counter += 1

    with open('output.csv', 'r') as file:
        num_rows = sum(1 for line in file)

    #test_acc, test_loss = compute_accuracy_loss(N_test, testSet, best_model)

    return train_loss_record, val_loss_record


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=3)
    parser.add_argument('--data_dir', default='')
    # resource/asnlib/publicdata/dev
    parser.add_argument('--learning_rate', default=0.146)
    parser.add_argument('--epoch_count', default=200)
    parser.add_argument('--minibatch_size', default=1100)
    parser.add_argument('--early_stopping_patience', default=25)
    parser.add_argument('--check_gradient', action="store_true", default=False,
                        help="Check the correctness of the gradient")
    parser.add_argument('--check_magnitude', action="store_true", default=False,
                        help="Check the magnitude of the gradient")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    val = get_parser()
    main(val)