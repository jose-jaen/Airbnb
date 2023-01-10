# Required packages
import tensorflow as tf
from keras.layers.core import Dense
from keras.models import Sequential
from keras import callbacks
from keras.utils import np_utils
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import uniform, quniform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from sklearn.preprocessing import StandardScaler


def nn_data():
    """ Data providing function for hyperopt.
    Separated from modeling function to avoid reloading data each evaluation

    - Output:
        - Scaled training, validation and test data
    """
    # Scaler is fit only to training data to avoid information leakage
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_valid = scaler.transform(X_valid)
    x_test = scaler.transform(X_test)
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def neural_network(x_train, y_train, x_valid, y_valid):
    """ Builds an Artificial Neural Network

    - Parameters:
        - x_train = Train feature matrix
        - y_train = Train target vector
        - x_valid = Validation feature matrix
        - y_valid = Validation target vector

    - Output:
        - val_loss = Root Mean Squared Error on validation data
        - STATUS_OK = Modeling status
        - model = Fitted ANN to training data
    """
    # Define layer weight initializer
    initial = initializers.HeNormal(seed=42)

    # Select optimizer to reduce loss
    optimizer = optimizers.Nadam(learning_rate=0.01,
                                 beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')

    # Define ANN architecture
    model = Sequential()

    # Set uniform distribution for neurons and regularization terms
    model.add(Dense({{quniform(120, 200, 1)}}, input_shape=(x_train.shape[1],),
                    activation='elu', kernel_initializer=initial))
    
    model.add(Dense({{quniform(950, 1050, 1)}}, activation='elu',
                    kernel_regularizer=regularizers.l1_l2(l1={{uniform(0.01, 0.2)}},
                                                          l2={{uniform(0.01, 0.07)}})))
    
    model.add(Dense({{quniform(950, 1050, 1)}}, activation='elu',
                    kernel_regularizer=regularizers.l1_l2(l1={{uniform(0.01, 0.2)}},
                                                          l2={{uniform(0.01, 0.07)}})))
    
    model.add(Dense({{quniform(950, 1050, 1)}}, activation='elu'))
                   
    
    model.add(Dense({{quniform(180, 290, 1)}}, activation='elu',
                    kernel_regularizer=regularizers.l1_l2(l1={{uniform(0.01, 0.3)}},
                                                          l2={{uniform(0.01, 0.3)}})))

    # Output layer with linear activation
    model.add(Dense(1, activation='linear'))

    # Compile model to reduce RMSE
    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  metrics=[RootMeanSquaredError()])

    # Store model results, choose batch size and epochs
    result = model.fit(x_train, y_train, verbose=2,
                       epochs=200, batch_size=int({{quniform(54, 60, 1)}}),
                       validation_data=(x_valid, y_valid))

    # Store evaluation metric
    val_loss = np.amin(result.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK, 'model': model}


def BNN(train, valid, test):
    """ Builds a Bayesian Neural Network

    - Parameters:
        - train = Dataset with training features and target
        - valid = Dataset with validation features and target
        - test = Dataset with test features and target

    - Output:
        - rmse = Root Mean Squared Error on test data
    """
    # Convert data to tensors
    x_train, train_y = torch.tensor(train[0]).float(), \
                       torch.tensor(train[1]).float()

    x_valid, valid_y = torch.tensor(valid[0]).float(), \
                       torch.tensor(valid[1]).float()

    x_test, test_y = torch.tensor(test[0]).float(), \
                     torch.tensor(test[1]).float()

    @variational_estimator
    class BayesianRegressor(nn.Module):
        # Define BNN architecture: hidden layers and neurons
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.blinear1 = BayesianLinear(input_dim, 124)
            self.blinear2 = BayesianLinear(124, 124)
            self.blinear3 = BayesianLinear(124, 75)
            self.blinear4 = BayesianLinear(75, output_dim)

        def forward(self, x):
            # Compute output Tensors with ReLU activation function
            x_ = self.blinear1(x)
            x_ = F.relu(x_)
            x_ = self.blinear2(x_)
            x_ = F.relu(x_)
            x_ = self.blinear3(x_)
            x_ = F.relu(x_)
            return self.blinear4(x_)

    def evaluate_regression(regressor, X, y, samples=100, std_multiplier=2):
        # Compute credible intervals for predictions
        preds = [regressor(X) for i in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)
        ci_upper = means + (std_multiplier*stds)
        ci_lower = means - (std_multiplier*stds)
        ic_acc = (ci_lower <= y)*(ci_upper >= y)
        ic_acc = ic_acc.float().mean()
        return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

    # Use GPU to speed up estimation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regressor = BayesianRegressor(x_train.shape[1], 1).to(device)

    # Define BNN optimizer and learning rate
    optimizer = optim.Adam(regressor.parameters(), lr=0.001)

    # Select cost function and minimization criterion
    criterion = torch.nn.MSELoss()

    # Specify training observations and batch size
    ds_train = torch.utils.data.TensorDataset(x_train, train_y.view(-1, 1))
    dataloader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=512, shuffle=True)

    # Specify validation data and batch size
    ds_valid = torch.utils.data.TensorDataset(x_valid, valid_y.view(-1, 1))
    dataloader_test = torch.utils.data.DataLoader(
        ds_valid, batch_size=512, shuffle=True)

    # Specify test data and batch size
    ds_test = torch.utils.data.TensorDataset(x_test, test_y.view(-1, 1))
    dataloader_test = torch.utils.data.DataLoader(
        ds_test, batch_size=512, shuffle=True)

    # Train BNN for some epochs
    iteration = 0
    for epoch in range(1000):
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            # Define loss
            loss = regressor.sample_elbo(inputs=datapoints.to(device),
                                         labels=labels.to(device), criterion=criterion,
                                         sample_nbr=3, complexity_cost_weight=1/train[0].shape[0])

            # Compute gradient of Tensors
            loss.backward()
            optimizer.step()

            # Periodically check training status of BNN
            iteration += 1
            if iteration % 100 == 0:
                ic_acc, under_ci_upper, over_ci_lower = evaluate_regression(regressor,
                                                                            x_test.to(device), test_y.to(device),
                                                                            samples=25, std_multiplier=3)

                print('CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}'.format(
                    ic_acc, under_ci_upper, over_ci_lower))
                print('Loss: {:.4f}'.format(loss))
    return np.sqrt(loss)
