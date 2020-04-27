# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# Creating the architecture of the Neural Network
# We're inheriting from nn.Module which we got from torch.nn
class SAE(nn.Module):
    def __init__(self, ):
        # Since Inheritance hence we are calling the super func
        # Use is to get inherited modules from the super class
        super(SAE, self).__init__()
        # First full connection denoted by fc1
        # no. features and no od nodes in hidden layers
        self.fc1 = nn.Linear(nb_movies, 20)
        # This will detect more features
        self.fc2 = nn.Linear(20, 10)
        # First part of decoding
        self.fc3 = nn.Linear(10, 20)
        # No. neurons should be same as no.of movies Also since Inputs = Outputs
        self.fc4 = nn.Linear(20, nb_movies)
        # Activation function
        self.activation = nn.Sigmoid()

    # It wil return vector of predicted ratings and then compare it with vector of input ratings
    def forward(self, x):
        # Encode the input vector of features to first hidden layers
        x = self.activation(self.fc1(x))  # x here is the input vector of features
        x = self.activation(self.fc2(x))
        # First part of decoding
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


sae = SAE()
# Criterion for loss function
criterion = nn.MSELoss()
# Optimizer to update the weights after each epoch
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        # Adding another dimension since pytorch expects batch of input vectors,so we're faking it
        input_vector = Variable(training_set(id_user)).unsqueeze(0)
        # Cloning since we will update the input vector but we need the input vector to check at the output layer
        target = input_vector.clone()
        # To save memory and check when user didn't rate the movie
        if torch.sum(target.data > 0) > 0:
            # Getting the ouput vector
            output = sae(input_vector)
            target.require_grad = False
            # User didn't rate the movies so they will not take part in weights updation
            output[target == 0] = 0
            # Computing the loss error
            loss = criterion(output, target)
            # The average of the error by considering only movies that have non-zero ratings
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        # Backward method will ensure do we need to increase the weights or decrease the weights(Decides the direction)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            # Optimizer decided the intensity of updation of weights
            optimizer.step()
        print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))


# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input_vector = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input_vector)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))
