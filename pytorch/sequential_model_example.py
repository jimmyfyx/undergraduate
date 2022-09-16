"""
Questions:
    1. plot of training loss and validation loss

"""

"""
Summary:
    - when evaluating the model, better to use with 'with torch.no_grad():', because this can reduce memory usage
      by preventing the model to trace gradients when going forward through the network.
    - model.eval() and model.train() have different function then 'with torch.no_grad():'. Their usage is to turn
      off/on dropout layers and batch_norm layers when going fowrad through the network, so it means that even with
      'with torch.no_grad():', it is necessary to apply model.eval() when actually evaluating the model.
    - all the functions that will be used in training can be organized in a single class.
    - usage of pandas and sklearn library in data processing before training. 
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

"""
read the dataset file using pandas
"""
# set the index of every row to datatime object from pandas
df = pd.read_csv('PJME_hourly.csv')

df = df.set_index(['Datetime'])
df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()
    
df = df.rename(columns={'PJME_MW': 'value'})

# generate values of previous n_lag timesteps for one datetime object
def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n
    
input_dim = 5
df_generated = generate_time_lags(df, input_dim)


"""
split into train, validation, and test datasets
"""
def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_generated, 'value', 0.2)
# size of X_train: (87219, 127); y_train: (87219, 1)
# size of X_val: (29073, 127); y_val: (29073, 1)
# size of X_test: (29074, 127); y_val: (29074. 1)


"""
scale the datasets
"""
scaler = MinMaxScaler()
X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)

# flip the order of every row (for the usage of RNN)
X_train_arr = np.flip(X_train_arr, 1)
X_val_arr = np.flip(X_val_arr, 1)
X_test_arr = np.flip(X_test_arr, 1)


"""
define hyperparameters
"""
batch_size = 64
sequence_length = X_train_arr.shape[1]
input_size = 1
hidden_size = 128
num_layers = 1
output_size = 1
num_epochs = 2
learning_rate = 0.001


"""
load datasets into dataloader
"""
train_features = torch.Tensor(X_train_arr.copy())
train_targets = torch.Tensor(y_train_arr)
val_features = torch.Tensor(X_val_arr.copy())
val_targets = torch.Tensor(y_val_arr)
test_features = torch.Tensor(X_test_arr.copy())
test_targets = torch.Tensor(y_test_arr)

train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)


"""
build model
"""
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTM(input_size, hidden_size, num_layers, output_size)


"""
train the model
"""
# define an optimization class that handling all the training steps
class Optimization:
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_losses = []
        self.batch_train_losses_overall = []
        self.val_losses = []
        self.batch_val_losses_overall = []
    
    def train_step(self, x, y):
        # x and y belong to a batch of training examples
        self.model.train()
        output = self.model(x)
        loss = self.loss_function(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, train_loader, val_loader, batch_size, num_epochs, sequence_length):
        for epoch in range(num_epochs):
            batch_losses = []
            for i, (x_batch, y_batch) in enumerate(train_loader):
                # reshape the matrix to (batch_size, sequence_length, n_features) to fit the input to LSTM
                x_batch = x_batch.view([batch_size, sequence_length, -1])
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
                self.batch_train_losses_overall.append(loss)

                if ((i + 1) % 10 == 0):
                    print(f"Epoch[{epoch + 1}/{num_epochs}], Batch[{i + 1}], Training Loss: {loss:.4f}")
            
            # calculate the average training loss for each epoch
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            print('\n')
            print('Training ends, start validation...')

            # validation process
            with torch.no_grad():
                batch_val_losses = []
                for j, (x_val, y_val) in enumerate(val_loader):
                    # reshape the matrix to (batch_size, sequence_length, n_features) to fit the input to LSTM
                    x_val = x_batch.view([batch_size, sequence_length, -1])
                    self.model.eval()
                    y_predicted = self.model(x_val)
                    val_loss = self.loss_function(y_predicted, y_val)
                    batch_val_losses.append(val_loss)
                    self.batch_val_losses_overall.append(val_loss)

                    if ((j + 1) % 10 == 0):

                        print(f"Epoch[{epoch + 1}/{num_epochs}], Batch[{j + 1}], Validation Loss: {val_loss:.4f}")
                
                # calculate the average validation loss for each epoch
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
            
            # get a summary of the training for each epoch
            print('\n')
            print('\n')
            print(f"Summary for Epoch[{epoch + 1}/{num_epochs}]: ")
            print(f"Epoch[{epoch + 1}/{num_epochs}] Training Loss: {training_loss:.4f}\t Validation Loss: {validation_loss:.4f}")
            print('\n')
            print('\n')

    def evaluate(self, test_loader, batch_size, sequence_length):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                # reshape the matrix to (batch_size, sequence_length, n_features) to fit the input to LSTM
                x_test = x_test.view([batch_size, sequence_length, -1])
                self.model.eval()
                y_predicted = self.model(x_test)
                predictions.append(y_predicted.detach().numpy())
                values.append(y_test.detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.batch_train_losses_overall, label="Training loss")
        plt.plot(self.batch_val_losses_overall, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


# initialize the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# execute training 
opt = Optimization(model=model, loss_function=loss_function, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, num_epochs=num_epochs, sequence_length=sequence_length)
opt.plot_losses()


"""
test and evaluate
"""
predictions, values = opt.evaluate(test_loader, batch_size=batch_size, sequence_length=sequence_length)

def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


df_result = format_predictions(predictions, values, X_test, scaler)
print(df_result)