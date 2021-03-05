import torch
from torch.autograd import Function
from torch.autograd import gradcheck
import pdb
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize, StandardScaler
from LINEAR_MSE import *
from sklearn.datasets import *
from torch.nn import Linear, Tanh ,MSELoss, Module, Sequential
from torch import optim


# description du contenu :
# reseau de neurones à trois couches

class Model(Module): # definition du reseau a 2 couches demandé
    def __init__(self,input_size,output_size, hiddenSize):
        super().__init__()
        self.linear1=Linear(input_size,hiddenSize)# premier lineaire
        self.linear2=Linear(hiddenSize, output_size)# second lineaire
        self.Tanh = Tanh()# le tanh
    def forward(self,x):
        yhat = self.linear2( self.Tanh( self.linear1(x) ) ) # linear -> tanh -> linear -> MSE
        return yhat



if __name__ == "__main__":
    # writer tensorboard
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M.%S")
    print("writer at : ",current_time)
    writer = SummaryWriter(f'./NN2/LOSS_{current_time}"')
    # load les données de test  et train
    X, y = load_boston(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1,1), test_size=0.33, random_state=42) # ? random state ?
    scaler = StandardScaler()
    # we fit the scaler on the train
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # data
    x_train_input = torch.from_numpy(X_train.astype('float32'))
    y_train_input = torch.from_numpy(y_train.astype('float32'))

    x_test_input = torch.from_numpy(X_test.astype('float32'))
    y_test_input = torch.from_numpy(y_test.astype('float32'))
    #define batch_size
    batch_size = 15
    # calculating the number of batches
    nb_batches = len(x_train_input)//batch_size if len(x_train_input)%batch_size == 0 \
    else len(x_train_input)//batch_size + 1
    # defining the model, with 100 hidden neurones

    # A decommenter pour la premiere version

    #model = Model(x_train_input.shape[1], y_train_input.shape[1], 100)
    #pour la version avec sequential
    input_size = x_train_input.shape[1] # taille de l'entrée
    hiddenSize = 100 # nombre de neurones en couche cachée
    output_size = y_train_input.shape[1] # taille de sortie
    model = Sequential(
          Linear(input_size,hiddenSize),
          Tanh(),
          Linear(hiddenSize, output_size)
        )

    # optimizer stochastic_gradient_descent
    optimizer = optim.SGD(model.parameters(), 0.001)
    # definition de la loss
    criterion = MSELoss()
    # mettre le grad a zero
    optimizer.zero_grad()
    # compteur
    cpt = 0

    for epoch in range(1000):
        # une permutation differente a chaque fois, ne pas utiliser la meme permutation
        rand_ind = torch.randperm(len(x_train_input))

        for batch in range(nb_batches):
            cpt +=1
            # load les données de train
            x = x_train_input[ batch *batch_size : (batch+1)*batch_size ]
            y = y_train_input[ batch *batch_size : (batch+1)*batch_size ]
            #zero grad de l'optimizer
            optimizer.zero_grad()
            # prediction
            yhat = model(x)
            # calcul de la loss
            loss = criterion(y, yhat)
            # backpropagate
            loss.backward()
            # faire la step du SGD
            optimizer.step()

            # test loss
            yhat_test = model(x_test_input)
            loss_test = criterion(y_test_input,yhat_test)

            # Sortie directe
            if batch % 1000 == 0:
                print(f"Itérations_train {cpt}: loss {loss}")
                print(f"Itérations_test {cpt}: loss {loss_test}")
        # by epoch loss
        # test loss : get the last test loss
        # train loss : on the whole train
        with torch.no_grad():
            yhat_epoch = model(x_train_input)
            loss_epoch = criterion(y_train_input,yhat_epoch)
        # log the result for the train
        writer.add_scalar('Loss/train', loss_epoch, epoch)
        # log the result for the test (on the whole test set)
        writer.add_scalar('Loss/test', loss_test, epoch)
