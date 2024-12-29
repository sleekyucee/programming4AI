#import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pickle

import warnings
warnings.filterwarnings("ignore")

class DataPreprocessor:
    def __init__(self, split_ratio=0.9, transpose=False):
        """
        Initialize the DataPreprocessing class.

        Parameters:
        - split_ratio: float, the ratio for train/val split (default is 0.9)
        - transpose: bool, whether to transpose features and target (default is False)
        """
        self.split_ratio = split_ratio
        self.transpose = transpose
        self.X_mean = None
        self.X_std = None

    def load(self, file_path):
        """
        Load the dataset from a CSV file.

        Parameters:
        - file_path: str, the path to the CSV file

        Returns:
        - df: pandas DataFrame
        """
        return pd.read_csv(file_path)

    def clean(self, df, index_col="Unnamed: 0", fill_column="Arrival_Delay_In_Minutes"):
        """
        Clean the dataset.

        Parameters:
        - df: pandas DataFrame, the dataset to clean
        - index_col: str, the column to set as index (default is "Unnamed: 0")
        - fill_column: str, the column to fill missing values (default is "Arrival_Delay_In_Minutes")

        Returns:
        - df: cleaned pandas DataFrame
        """
        df = df.set_index(index_col)
        df.index.name = "Index"
        df.columns = df.columns.str.replace(" ", "_", regex=True).str.title()

        if df[fill_column].isnull().sum() > 0:
            df[fill_column] = df[fill_column].fillna(df[fill_column].mean())

        return df

    def process_train(self, df, target_column, categorical_columns):
        """
        Process the training dataset.

        Parameters:
        - df: pandas DataFrame, the dataset to process
        - target_column: str, the name of the target column
        - categorical_columns: list, the categorical columns to map to numeric

        Returns:
        - X_train, X_val, y_train, y_val: processed features and target
        """
        #map categorical columns to numeric
        for col in categorical_columns:
            df[col] = df[col].map({k: i for i, k in enumerate(df[col].unique())})

        #split into features and target
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values

        #split into train and validation sets
        split = int(self.split_ratio * len(y))
        X_train, X_val = np.split(X, [split])
        y_train, y_val = np.split(y, [split])

        #standardize features
        self.X_mean = np.mean(X_train, axis=0, keepdims=True)
        self.X_std = np.std(X_train, axis=0, keepdims=True)

        X_train = (X_train - self.X_mean) / (self.X_std + 1e-8)
        X_val = (X_val - self.X_mean) / (self.X_std + 1e-8)

        #transpose if required
        if self.transpose:
            X_train = X_train.T
            X_val = X_val.T
            y_train = y_train.reshape(1, -1)
            y_val = y_val.reshape(1, -1)

        return X_train, X_val, y_train, y_val

    def process_test(self, df, target_column, categorical_columns):
        """
        Process the test dataset.

        Parameters:
        - df: pandas DataFrame, the test dataset to process
        - target_column: str, the name of the target column
        - categorical_columns: list, the categorical columns to map to numeric

        Returns:
        - X_test, y_test: processed features and target
        """
        #map categorical columns to numeric
        for col in categorical_columns:
            df[col] = df[col].map({k: i for i, k in enumerate(df[col].unique())})

        #split into features and target
        X_test = df.drop(target_column, axis=1).values
        y_test = df[target_column].values

        #standardize features using training mean and std
        X_test = (X_test - self.X_mean) / (self.X_std + 1e-8)

        #transpose if required
        if self.transpose:
            X_test = X_test.T
            y_test = y_test.reshape(1, -1)

        return X_test, y_test

#save class names
class_names = ["Neutral or Dissatisfied", "Satisfied"]

def sigmoid_forward(x):
    sig_forward = 1/(1 + np.exp(-x))

    return sig_forward

def sigmoid_backward(sig_forward):
    sig_derivative = sig_forward * (1 - sig_forward)

    return sig_derivative

def relu_forward(x):
    rel_forward = np.maximum(0, x)

    return rel_forward

def relu_backward(x):
    rel_derivative = np.where(x > 0, 1, 0)

    return rel_derivative

def softmax_forward(x):
    #subtract max from x for numerical stability
    x -= np.max(x, axis=0, keepdims=True)

    exp_x_i = np.exp(x)
    sum_exp_x_j = np.sum(exp_x_i, axis=0, keepdims=True)
    soft_forward = exp_x_i/sum_exp_x_j
    return soft_forward

def softmax_backward(x):
    #retrieve classes and samples
    n, m = x.shape
    jacobians = np.zeros((m, n, n))  #shape: (n_samples, n_classes, n_classes)

    for sample_idx in range(m):
        #extract softmax probabilities for a single sample
        y = x[:, sample_idx]

        #compute the Jacobian matrix for this sample
        jacobian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                #diagonal case
                if i == j:
                    jacobian[i, j] = y[i] * (1 - y[i])
                #off-diagonal case
                else:
                    jacobian[i, j] = -y[i] * y[j]

        #store the Jacobian matrix
        jacobians[sample_idx] = jacobian

    return jacobians

def inverted_dropout_forward(x, p, activation_function):
    #set keep probability
    if not 0 < p <= 1:
        raise ValueError("Probability p must be between 0 and 1.")

    a = activation_function(x)

    #create mask
    mask = (np.random.rand(*a.shape) < p) / p

    #apply mask
    a = a * mask

    return a, mask

def inverted_dropout_backward(dA, mask):
    dA = dA * mask

    return dA

class NeuralNetwork:
    def __init__(self, layer_dims, activations, dropout_probs=1.0, regularizer=None, reg_lambda=0.001, optimizer="sgd", seed=None):
        if seed is not None:
            np.random.seed(seed)
        assert len(layer_dims) == len(activations) + 1, "Number of activations must match the number of layers minus one."
        self.layer_dims = layer_dims
        self.activations = activations
        self.dropout_probs = dropout_probs if dropout_probs is not None else [1.0] * (len(layer_dims) - 1)
        assert len(self.dropout_probs) == len(self.layer_dims) - 1, "Number of dropout probabilities must match number of hidden layers"
        self.regularizer = regularizer
        self.reg_lambda = reg_lambda
        self.params = self._initialize_weights()

    def _initialize_weights(self):
        params = {}
        for l in range(1, len(self.layer_dims)):
            params[f"W{l}"] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * np.sqrt(2 / self.layer_dims[l - 1])
            params[f"b{l}"] = np.zeros((self.layer_dims[l], 1))
        return params

    def forward(self, X, is_training=True):
        activations = [X]
        caches = []
        A = X
        for i in range(1, len(self.layer_dims)):
            W, b = self.params[f"W{i}"], self.params[f"b{i}"]
            Z = np.dot(W, A) + b
            if self.activations[i - 1] == "sigmoid":
                activation_function = sigmoid_forward
                A = activation_function(Z)
            elif self.activations[i - 1] == "relu":
                activation_function = relu_forward
                A = activation_function(Z)
            elif self.activations[i - 1] == "softmax":
                activation_function = softmax_forward
                A = activation_function(Z)
            else:
                raise ValueError("Unsupported activation function.")

            if is_training and self.dropout_probs[i - 1] < 1.0:
                A, mask = inverted_dropout_forward(Z, self.dropout_probs[i - 1], activation_function)
                caches.append((Z, mask))
            else:
                caches.append(Z)

            activations.append(A)
            if self.activations[i - 1] == "softmax":
                assert np.allclose(np.sum(A, axis=0), 1, atol=1e-6), "Softmax outputs do not sum to 1."

        return A, activations, caches

    def backward(self, X, Y, activations, caches):
        grads = {}
        m = X.shape[1]
        L = len(self.layer_dims) - 1

        #output layer
        activation_output = self.activations[-1]
        if activation_output == "softmax":
            dZ = activations[-1] - Y
        elif activation_output == "sigmoid":
            dZ = (activations[-1] - Y) * activations[-1] * (1 - activations[-1])
        else:
            raise ValueError("Unsupported activation function for output layer.")

        dW = (1 / m) * np.dot(dZ, activations[-2].T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if self.regularizer == "L2":
            dW += (self.reg_lambda / m) * self.params[f"W{L}"]
        elif self.regularizer == "L1":
            dW += (self.reg_lambda / m) * np.sign(self.params[f"W{L}"])

        grads[f"dW{L}"] = dW
        grads[f"db{L}"] = db
        dA = np.dot(self.params[f"W{L}"].T, dZ)

        #remaining layers
        for l in reversed(range(1, L)):
            cache = caches[l - 1]
            if isinstance(cache, tuple):  #dropout applied
                Z, mask = cache
                dA = inverted_dropout_backward(dA, mask)

            Z = cache if not isinstance(cache, tuple) else cache[0]
            if self.activations[l - 1] == "relu":
                dZ = dA * relu_backward(Z)
            elif self.activations[l - 1] == "sigmoid":
                dZ = dA * sigmoid_backward(Z)
            elif self.activations[l - 1] == "softmax":
                dZ = dA * softmax_backward(Z)
            else:
                raise ValueError("Unsupported activation function.")

            dW = (1 / m) * np.dot(dZ, activations[l - 1].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if self.regularizer == "L2":
                dW += (self.reg_lambda / m) * self.params[f"W{l}"]
            elif self.regularizer == "L1":
                dW += (self.reg_lambda / m) * np.sign(self.params[f"W{l}"])

            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db
            dA = np.dot(self.params[f"W{l}"].T, dZ)

        return grads

    def compute_loss(self, Y, Y_pred, output_activation="softmax"):
        m = Y.shape[1]
        assert Y_pred.shape == Y.shape, f"Y_pred shape mismatch: expected {Y.shape}, got {Y_pred.shape}"
        if output_activation == "softmax":
            cross_entropy_loss = -np.mean(np.sum(Y * np.log(Y_pred + 1e-8), axis=0))
        elif output_activation == "sigmoid":
            cross_entropy_loss = -np.mean(Y * np.log(Y_pred + 1e-8) + (1 - Y) * np.log(1 - Y_pred + 1e-8))
        else:
            raise ValueError("Unsupported output activation function. Use 'softmax' or 'sigmoid'.")
        reg_loss = 0
        if self.regularizer == "L2":
            for i in range(1, len(self.layer_dims)):
                reg_loss += np.sum(np.square(self.params[f"W{i}"]))
            reg_loss = (self.reg_lambda / (2 * m)) * reg_loss
        elif self.regularizer == "L1":
            for i in range(1, len(self.layer_dims)):
                reg_loss += np.sum(np.abs(self.params[f"W{i}"]))
            reg_loss = (self.reg_lambda / m) * reg_loss
        return cross_entropy_loss + reg_loss

    def update_weights(self, X, Y, learning_rate, grads=None, optimizer="none", epochs=1, batch_size=None):
        clip_threshold = 10  #threshold for gradient clipping
        m = X.shape[1]  #number of samples

        if optimizer == "sgd":
            for i in range(1, len(self.layer_dims)):
                self.params[f"W{i}"], self.params[f"b{i}"] = sgd_optimizer(
                    X, Y,
                    self.params[f"W{i}"], self.params[f"b{i}"],
                    learning_rate, epochs
                )

        elif optimizer == "sgd_mini_batch":
            for i in range(1, len(self.layer_dims)):
                self.params[f"W{i}"], self.params[f"b{i}"] = sgd_mini_batch(
                    X, Y,
                    self.params[f"W{i}"], self.params[f"b{i}"],
                    learning_rate, epochs, batch_size
                )

        else:  #batch Gradient Descent (default if optimizer="none")
            for epoch in range(epochs):
                #forward pass to compute predictions
                Y_pred, activations, caches = self.forward(X, is_training=True)

                #backward pass to compute gradients
                grads = self.backward(X, Y, activations, caches)

                #update weights and biases for all layers
                for i in range(1, len(self.layer_dims)):
                    dW = grads[f"dW{i}"]
                    db = grads[f"db{i}"]

                    #clip gradients
                    for grad_key in [f"dW{i}", f"db{i}"]:
                        grad = grads[grad_key]
                        grad_norm = np.linalg.norm(grad)
                        if grad_norm > clip_threshold:
                            grads[grad_key] *= (clip_threshold / grad_norm)

                    #regularization
                    if self.regularizer == "L2":
                        dW += (self.reg_lambda / m) * self.params[f"W{i}"]
                    elif self.regularizer == "L1":
                        dW += (self.reg_lambda / m) * np.sign(self.params[f"W{i}"])

                    #gradient update
                    self.params[f"W{i}"] -= learning_rate * dW
                    self.params[f"b{i}"] -= learning_rate * db

    def compute_accuracy(self, Y, Y_pred):
        """Compute accuracy by comparing true labels with predicted labels."""
        if self.activations[-1] == "softmax":
            predictions = np.argmax(Y_pred, axis=0)
            true_labels = np.argmax(Y, axis=0)
        elif self.activations[-1] == "sigmoid":
            predictions = (Y_pred > 0.5).astype(int)
            true_labels = Y.astype(int)
        else:
            raise ValueError("Unsupported output activation for accuracy calculation.")

        return np.mean(predictions == true_labels)


    def compute_weight_norm(self):
        """Compute the Frobenius norm of all weights."""
        total_norm = 0
        for i in range(1, len(self.layer_dims)):
            total_norm += np.linalg.norm(self.params[f"W{i}"]) ** 2
        return np.sqrt(total_norm)

    def train(self, X_train, Y_train, X_val, Y_val, epochs, learning_rate, batch_size=None, decay_rate=0.0, optimizer="none"):
        m = X_train.shape[1]
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'train_acc': [],
            'val_acc': [],
            'weight_norm': []
        }
        current_lr = learning_rate

        for epoch in range(epochs):
            epoch_loss = 0

            if optimizer == "none":  #batch gradient descent
                Y_pred, activations, caches = self.forward(X_train, is_training=True)
                train_loss = self.compute_loss(Y_train, Y_pred, output_activation=self.activations[-1])
                epoch_loss = train_loss  #full batch contributes to the epoch loss

                grads = self.backward(X_train, Y_train, activations, caches)
                self.update_weights(X_train, Y_train, learning_rate, grads=grads, optimizer="none", epochs=1)

            elif optimizer == "sgd":  #stochastic gradient descent
                for i in range(m):  #loop over individual samples
                    xi = X_train[:, i:i+1]
                    yi = Y_train[:, i:i+1]

                    Y_pred, activations, caches = self.forward(xi, is_training=True)
                    train_loss = self.compute_loss(yi, Y_pred, output_activation=self.activations[-1])
                    epoch_loss += train_loss

                    self.update_weights(xi, yi, learning_rate, optimizer="sgd", epochs=1)

                epoch_loss /= m  #average loss over samples

            elif optimizer == "sgd_mini_batch":  #mini-Batch Gradient Descent
                if batch_size is None:
                    raise ValueError(f"batch_size must be specified for mini-batch '{optimizer}' optimizer")

                perm = np.random.permutation(m)  #shuffle data
                X_train, Y_train = X_train[:, perm], Y_train[:, perm]
                num_minibatches = (m + batch_size - 1) // batch_size  #calculate the number of mini-batches

                for i in range(num_minibatches):
                    start = i * batch_size
                    end = min(start + batch_size, m)
                    X_batch = X_train[:, start:end]
                    Y_batch = Y_train[:, start:end]

                    Y_pred, activations, caches = self.forward(X_batch, is_training=True)
                    train_loss = self.compute_loss(Y_batch, Y_pred, output_activation=self.activations[-1])
                    epoch_loss += train_loss

                    self.update_weights(X_batch, Y_batch, learning_rate, optimizer="sgd_mini_batch", epochs=1, batch_size=batch_size)

                epoch_loss /= num_minibatches  #average loss over mini-batches

            #record epoch metrics
            history['train_loss'].append(epoch_loss)

            #validation metrics
            Y_val_pred, _, _ = self.forward(X_val, is_training=False)
            val_loss = self.compute_loss(Y_val, Y_val_pred, self.activations[-1])
            history['val_loss'].append(val_loss)

            train_acc = self.compute_accuracy(Y_train, self.forward(X_train, is_training=False)[0])
            val_acc = self.compute_accuracy(Y_val, Y_val_pred)
            weight_norm = self.compute_weight_norm()

            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['weight_norm'].append(weight_norm)

            #update learning rate with decay
            if decay_rate > 0.0:
                current_lr *= np.exp(-decay_rate)
            history['learning_rate'].append(current_lr)

            #print metrics for the epoch
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {epoch_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}"
                  )

        return history


    def model_summary(self):
        total_params = 0
        print("Layer\tInput Dim\tOutput Dim\tParameters")
        print("=" * 40)

        for i in range(1, len(self.layer_dims)):
            input_dim = self.layer_dims[i - 1]
            output_dim = self.layer_dims[i]
            num_weights = input_dim * output_dim
            num_biases = output_dim
            layer_params = num_weights + num_biases
            total_params += layer_params

            print(f"{i}\t{input_dim}\t\t{output_dim}\t\t{layer_params}")

        print("=" * 40)
        print(f"Total Trainable Parameters: {total_params}")
        return total_params


    def predict(self, X):
        Y_pred, _, _ = self.forward(X, is_training=False)
        return Y_pred

    def plot_loss_and_accuracy(self, history):
        epochs = range(1, len(history['train_loss']) + 1)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        #plot loss on the left y-axis
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color='tab:red')
        ax1.plot(epochs, history['train_loss'], label="Train Loss", color='tab:red', linestyle='--')
        ax1.plot(epochs, history['val_loss'], label="Val Loss", color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc="upper left")
        ax1.grid(True)

        #create a twin axis for accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color='tab:blue')
        ax2.plot(epochs, history['train_acc'], label="Train Accuracy", color='tab:blue', linestyle='--')
        ax2.plot(epochs, history['val_acc'], label="Val Accuracy", color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.legend(loc="upper right")

        plt.title("Loss and Accuracy Over Epochs")
        plt.show()

    def plot_weight_norms(self, history):
        epochs = range(1, len(history['weight_norm']) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['weight_norm'], label="Weight Norms", color='tab:green')
        plt.xlabel("Epochs")
        plt.ylabel("Weight Norm")
        plt.title("Weight Norms Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_learning_rate(self, history):
        #plot learning rate over epochs
        epochs = range(1, len(history['learning_rate']) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['learning_rate'], color='red', label="Learning Rate")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Decay")
        plt.grid(True)
        plt.show()

    def evaluate(self, X, Y, title="Confusion Matrix"):
        Y_pred = self.predict(X)
        if self.activations[-1] == "softmax":
            Y_pred_labels = np.argmax(Y_pred, axis=0)
            Y_true_labels = np.argmax(Y, axis=0)
            accuracy = np.mean(Y_pred_labels == Y_true_labels)
            cm = confusion_matrix(Y_true_labels, Y_pred_labels)

            #normalize confusion matrix to get percentages
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(title)
            plt.show()

        elif self.activations[-1] == "sigmoid":
            Y_pred_labels = (Y_pred > 0.5).astype(int)
            Y_true_labels = Y.astype(int)
            accuracy = np.mean(Y_pred_labels == Y_true_labels)
            cm = confusion_matrix(Y_true_labels.flatten(), Y_pred_labels.flatten())

            #normalize confusion matrix to get percentages
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(title)
            plt.show()

        else:
            raise ValueError("Unsupported output activation for evaluation. Use 'softmax' or 'sigmoid'.")

        return accuracy

#main script
if __name__ == "__main__":
    #initialize preprocessor
    preprocessor = DataPreprocessor(split_ratio=0.9, transpose=True)

    #load training data
    url = "https://raw.githubusercontent.com/sleekyucee/programming4AI/main/Datasets/train.csv"
    train_data = preprocessor.load(url)

    #clean and preprocess training data
    train_df = preprocessor.clean(train_data)
    X_train, X_val, y_train, y_val = preprocessor.process_train(
        train_df, 
        target_column="Satisfaction", 
        categorical_columns=["Gender", "Customer_Type", "Type_Of_Travel", "Class", "Satisfaction"]
    )

    #load test data
    url = "https://raw.githubusercontent.com/sleekyucee/programming4AI/main/Datasets/test.csv"
    test_data = preprocessor.load(url)

    #clean and preprocess test data
    test_df = preprocessor.clean(test_data)
    X_test, y_test = preprocessor.process_test(
        test_df, 
        target_column="Satisfaction", 
        categorical_columns=["Gender", "Customer_Type", "Type_Of_Travel", "Class", "Satisfaction"]
    )

#model configuration
layer_dims = [X_train.shape[0], 512, 256, 128, y_train.shape[0]]
activations = ["relu", "relu", "relu", "sigmoid"]
dropout_probs = [1.0, 0.8, 0.8, 1.0]  #dropout applied to hidden layers
regularizer = "L2"
reg_lambda = 0.001
optimizer = "sgd_mini_batch"
seed = 36

#initialize model
model = NeuralNetwork(
    layer_dims,
    activations,
    dropout_probs,
    regularizer=regularizer,
    optimizer=optimizer,
    reg_lambda=reg_lambda,
    seed=seed)

#set epochs and batch size
epochs = 100
batch_size = 32

#set learning and decay rates
learning_rate = 0.01
decay_rate = 0.005

#train model
history = model.train(X_train,
                         y_train,
                         X_val,
                         y_val,
                         epochs=epochs,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         decay_rate=decay_rate)
#retrain model a further 200 epochs
epochs = 200

#increase learning rate and decay rate
learning_rate = 0.1
decay_rate = 0.01

#train model
history = model.train(X_train,
                         y_train,
                         X_val,
                         y_val,
                         epochs=epochs,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         decay_rate=decay_rate)
#retrain model  a further 200 epochs
epochs = 200

#train model
history = model.train(X_train,
                         y_train,
                         X_val,
                         y_val,
                         epochs=epochs,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         decay_rate=decay_rate)

#retrain model a further 200 epochs
epochs = 200
batch_size = 32

#train model
history = model.train(X_train,
                         y_train,
                         X_val,
                         y_val,
                         epochs=epochs,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         decay_rate=decay_rate)

#get model summary
model.model_summary()

#plot losses and accuracies
model.plot_loss_and_accuracy(history)

#get predictions on X_test
y_pred7 = model.predict(X_test)
y_pred7 = (y_pred7 > 0.5).astype(int)
y_pred7

#evaluate model 7
accuracy7 = model.evaluate(X_test, y_test)
print(f"\nModel 7 - Test Acc: {accuracy7 * 100:.2f}%")