from random import shuffle
import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.nn import functional as F
from convLSTM import ConvLSTM
from convLSTM import ConvLSTMCell
from data.MovingMNIST import MovingMNIST


"""
Define Hyperparameters 
"""
learning_rate = 0.001
batch_size = 20
num_epochs = 150
num_steps_past = 10
num_steps_ahead = 10 # in moving minist, the sequence length is 20
num_hidden_dim = 16
input_channel = 1 # moving minist dataset only has 1 channel per image (black and white)

path = './data/mnist'

"""
Dataset and Dataloader 
"""
train_data = MovingMNIST(train=True, data_root=path, seq_len=num_steps_past + num_steps_ahead, image_size=64, deterministic=True, num_digits=2)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = MovingMNIST(train=False, data_root=path, seq_len=num_steps_past + num_steps_ahead, image_size=64, deterministic=True, num_digits=2)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


"""
Build the model from ConvLSTMCell
"""
class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()
        self.encoder_1 = ConvLSTMCell(input_dim=in_chan, hidden_dim=nf, kernel_size=(3, 3), bias=True)
        self.encoder_2 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True)
        self.decoder_1 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True)
        self.decoder_2 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True)
        self.cnn = nn.Conv3d(in_channels=nf, out_channels=1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
    
    def forward(self, x, future_seq=0, hidden_state=None):
        b_size, seq_len, n_channel, height, width = x.shape
        h_t, c_t = self.encoder_1.init_hidden(batch_size=b_size, image_size=(height, width))
        h_t2, c_t2 = self.encoder_2.init_hidden(batch_size=b_size, image_size=(height, width))
        h_t3, c_t3 = self.decoder_1.init_hidden(batch_size=b_size, image_size=(height, width))
        h_t4, c_t4 = self.decoder_2.init_hidden(batch_size=b_size, image_size=(height, width))

        outputs = self.encoder_decoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)
        return outputs


    def encoder_decoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):
        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        # print(outputs)
        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.cnn(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

model = EncoderDecoderConvLSTM(nf=num_hidden_dim, in_chan=input_channel)



"""
train the model
"""
# define an optimization class that handling all the training steps
class Optimization:
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_losses = [] # store training loss for each epoch
    

    def train_step(self, x, y):
        # x and y belong to a batch of training examples
        self.model.train()
        output = self.model(x, num_steps_ahead)
        loss = self.loss_function(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def train(self, train_loader, batch_size, num_epochs):
        for epoch in range(num_epochs):
            batch_losses = []
            for i, batch in enumerate(train_loader):
                """
                batch has size (batch_size, sequence_length=20, height=64, width=64, n_channels=1)
                """
                x, y = batch[:, 0:num_steps_past, :, :, :], batch[:, num_steps_past:, :, :, :]
                x = x.permute(0, 1, 4, 2, 3)
                y = y.squeeze()

                loss = self.train_step(x, y)
                batch_losses.append(loss)

                if ((i + 1) % 10 == 0):
                    print(f"Epoch[{epoch + 1}/{num_epochs}], Batch[{i + 1}], Training Loss: {loss:.4f}")
            
            # calculate the average training loss for each epoch
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

        average_train_loss = np.mean(self.train_losses)
        print('\n')
        print(f"Training finishes, average training loss:{average_train_loss}")
        print('\n')


    def evaluate(self, test_loader):
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                """
                batch has size (batch_size, sequence_length=20, height=64, width=64, n_channels=1)
                """
                
                # generate output images every 10 batch step
                if i % 10 == 0:
                    x_test, y_test = batch[:, 0:num_steps_past, :, :, :], batch[:, num_steps_past:, :, :, :]
                    x_test = x_test.permute(0, 1, 4, 2, 3)
                    y_test = y_test.permute(0, 1, 4, 2, 3)
                    self.model.eval()
                    y_predicted = self.model(x_test, num_steps_ahead)

                    # output one sequence example among (batch_size) sequence examples
                    y_predicted = y_predicted[0]
                    y_predicted = y_predicted.permute(1, 0, 2, 3)
                    y_test = y_test[0]
                    
                    predicted_seq = y_predicted[0]
                    j = 1
                    while j <= 9:
                        predicted_seq = torch.cat((predicted_seq, y_predicted[j]), 2)
                        j = j + 1
                    
                    test_seq = y_test[0]
                    j = 1
                    while j <= 9:
                        test_seq = torch.cat((test_seq, y_test[j]), 2)
                        j = j + 1
                    sequence_output = torch.cat((predicted_seq, test_seq), 2)
                    save_image(sequence_output, f'./MINIST_output/batch{i}_output.png')
        
        print('\n')
        print("Evaluation Finishes")
        print('\n')




# initialize the loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# execute training 
opt = Optimization(model=model, loss_function=loss_function, optimizer=optimizer)
opt.train(train_loader, batch_size=batch_size, num_epochs=num_epochs)
opt.evaluate(test_loader=test_loader)