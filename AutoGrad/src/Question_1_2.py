# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck
import pdb
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize
from sklearn.preprocessing import StandardScaler
from LINEAR_MSE import *
from sklearn.datasets import *

# description du contenu
# descente de gradient batch - données : Boston Housing

if __name__ == "__main__":

    # load, split and scale the data
    X, y = load_boston(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1,1), test_size=0.33, random_state=42)
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

    # parameters to optimize
    w = torch.randn((x_train_input.shape[1], y_train_input.shape[1]), requires_grad = True, dtype=torch.float32)
    b = torch.randn((1,y_train_input.shape[1]), requires_grad = True, dtype=torch.float32)


    # hyperparameters
    epsilon = 0.1
    # logs
    writer = SummaryWriter()

    # gradient descent
    for n_iter in range(1000):
        ## Calcul du forward (loss)
        yhat = Linear.forward(x_train_input, w, b)
        loss = MSE.forward(y_train_input,yhat)
        loss.backward()

        # vaeurs du test
        yhat_test = Linear.forward(x_test_input, w, b)
        loss_test = MSE.forward(y_test_input,yhat_test)


        # logs
        writer.add_scalar('Loss/train', loss, n_iter)
        writer.add_scalar('Loss/test', loss_test, n_iter)

        # sortie directe
        print(f"Itérations_train {n_iter}: loss {loss}")
        print(f"Itérations_test {n_iter}: loss {loss_test}")

        # update the parameters
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad

        # grad to zero
        w.grad.zero_()
        b.grad.zero_()
