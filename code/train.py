# pylint: skip-file
'''
@author: Guillem Amat
'''

#Local Packages
from utils import load_word2int
from neural_net import LSTM
import preprocessing

#Usual Packages
import pandas as pd
import numpy as np
import os

#Torch Packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def train(model: nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          test_loader: torch.utils.data.DataLoader, 
          batch: int, epochs: int = 5) -> nn.Module:
    
    # Initializing variables
    train_loss = 0
    test_loss  = 0
    model_loss = np.inf

    # To store training results
    train_history  = []
    test_history   = []
    train_accuracy = []
    test_accuracy  = []

    for epoch in range(epochs):

        # Defining training configuration
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        #Reset Loss values and init hidden layer
        train_loss = 0
        test_loss  = 0
        hidden = model.init_hidden(batch)

        model.train()
        for _, data in enumerate(train_loader):
                        
            optimizer.zero_grad()
            
            X_train, y_train = data
            X_train, y_train = Variable(torch.squeeze(X_train)), Variable(torch.squeeze(y_train))

            #Forward pass
            hidden = tuple([each.data for each in hidden])
            y_hat, hidden = model(X_train, hidden)
            
            #Calculating loss and backpropagating
            loss  = criterion(y_hat, y_train)
            loss.backward()
            
            #Clipping Gradients and taking a step
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            #Storing loss results
            train_loss += loss.item() 
            train_history.append(loss.item())
            
            #Calculating Accuracy
            predictions = torch.round(y_hat.squeeze())
            num_correct = predictions.eq(y_train.float().view_as(predictions))
            correct     = np.squeeze(num_correct.numpy())
            accuracy    = np.sum(correct)/batch
            train_accuracy.append(accuracy)
        
        hidden = model.init_hidden(batch)
        
        #print('test')
        model.eval()
        for _, data in enumerate(test_loader):
            
            X_test, y_test = data
            X_test, y_test = torch.squeeze(X_test), torch.squeeze(y_test)
            
            hidden = tuple([each.data for each in hidden])
            y_hat, hidden = model(X_test, hidden)
            
            #Checking Loss
            loss = criterion(y_hat, y_test)
            test_loss += loss.item()
            test_history.append(loss.item())
            
            #Calculating Accuracy
            predictions = torch.round(y_hat.squeeze())
            num_correct = predictions.eq(y_test.float().view_as(predictions))
            correct     = np.squeeze(num_correct.numpy())
            accuracy    = np.sum(correct)/batch
            test_accuracy.append(accuracy)
            
        
        if test_loss < model_loss:
            PATH = os.path.join(os.getcwd(), '..', 'models', 'Sentiment_LSTM_2')
            torch.save(model, PATH)
            PATH = os.path.join(os.getcwd(), '..', 'models', 'Sentiment_LSTM_dictionary_2')
            torch.save(model.state_dict(), PATH) 
            model_loss = test_loss  
        
        train_loss = str(train_loss*batch/len(train_loader.dataset))[:4]
        test_loss  = str(test_loss*batch/len(test_loader.dataset))[:4]      
        
        string = f'''| Epoch: {epoch + 1}   | Train Loss: {train_loss} | Test Loss: {str(test_loss)[:4]} |'''
        print(string); print('-'*len(string))
            

def main():

    # Load word dictionary
    word2int = load_word2int(os.path.join(os.getcwd(), '..', 'data', 'word2int'))
    vocabulary = len(word2int) + 1

    # Creating a model instance
    lstm = LSTM(vocabulary, n_output=1, n_embedding=500, n_hidden=4, n_layers=1)

    # preprocess data
    X_train, y_train, X_test, y_test = preprocessing.main()

    #Creating torch Tensors
    data_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).long(), torch.from_numpy(np.array(y_train)).float())
    data_test  = torch.utils.data.TensorDataset(torch.from_numpy(X_test).long(), torch.from_numpy(np.array(y_test)).float())

    # Creating DataLoaders
    batch = 5000
    train_loader = torch.utils.data.DataLoader(data_train, shuffle=True, batch_size=batch, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(data_test, shuffle= True, batch_size=batch, drop_last=True)

    # Formatting
    print('\n')
    print(' '*17 + 'Results Summary', '-'*51, sep='\n')

    #Training Model
    train(model = lstm, train_loader=train_loader, test_loader=test_loader, epochs = 2, batch = batch)

    # Formatting
    print('\n', 'Done!', '\n')


if __name__ == "__main__":
    main()