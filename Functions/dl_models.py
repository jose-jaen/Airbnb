# Required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


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
            self.blinear2 = BayesianLinear(124, 75)
            self.blinear3 = BayesianLinear(75, output_dim)

        def forward(self, x):
            # Compute output Tensors with ReLU activation function
            x_ = self.blinear1(x)
            x_ = F.relu(x_)
            x_ = self.blinear2(x_)
            x_ = F.relu(x_)
            return self.blinear3(x_)

    def evaluate_regression(regressor, X, y, samples=100, std_multiplier=2):
        # Compute credible intervals for predictions
        preds = [regressor(X) for i in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)
        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)
        ic_acc = (ci_lower <= y) * (ci_upper >= y)
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
                                         sample_nbr=3, complexity_cost_weight=1 / train[0].shape[0])

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