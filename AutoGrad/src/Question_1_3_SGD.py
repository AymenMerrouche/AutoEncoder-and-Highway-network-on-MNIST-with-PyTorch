import torch
from torch.autograd import Function
from torch.autograd import gradcheck
import pdb
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize, StandardScaler
from LINEAR_MSE import *
from sklearn.datasets import *


# description du contenu :
# algorihtme de descente de gradient stochastic

# mini batch gradient descent
def stochastic_gradient_descent(x_train_input, x_test_input, y_train_input, y_test_input, Model = Linear,\
                                Loss = MSE, nb_epochs = 1000, batch_size = 1, learning_rate = 0.1,\
                                writer = SummaryWriter(), verbose = True):
    # parameters to optimize
    w = torch.randn((x_train_input.shape[1], y_train_input.shape[1]), requires_grad = True, dtype=torch.float32)
    b = torch.randn((1,y_train_input.shape[1]), requires_grad = True, dtype=torch.float32)



    # batch size div + 1 (last one may be smaller)
    nb_batches = len(x_train_input)//batch_size if len(x_train_input)%batch_size == 0 \
    else len(x_train_input)//batch_size + 1

    # main loop

    # how many times we changed the gradient
    cpt = 0
    for n_iter in range(nb_epochs):
        # artefact to create batches
        randomised_indexes = torch.randperm(len(x_train_input))
        for batch_number in range(nb_batches):

            # we are going to modify the gradient
            cpt += 1

            # get current batch train (the selection is already randomized using the previous artefact)
            batch_x = x_train_input[randomised_indexes[batch_number*batch_size:(batch_number+1)*batch_size]]
            batch_y = y_train_input[randomised_indexes[batch_number*batch_size:(batch_number+1)*batch_size]]


            ## Compute the forward (loss) and propagate
            yhat = Linear.forward(batch_x, w, b)
            loss = MSE.forward(batch_y,yhat)
            loss.backward()
            # Mise à jour des paramètres du modèle
            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            # remise à zero du gradient (additif ici)
            w.grad.zero_()
            b.grad.zero_()
        # after each batch compute the loss on the train and on the test
        with torch.no_grad():
            yhat_train = Linear.forward(x_train_input, w, b)
            yhat_test = Linear.forward(x_test_input, w, b)
            loss_epoch_train = MSE.forward(yhat_train, y_train_input)
            loss_epoch_test = MSE.forward(yhat_test, y_test_input)
        # log by epoch loss (mean)
        writer.add_scalar('Epoch/epoch_loss_Train',loss_epoch_train, n_iter)
        writer.add_scalar('Epoch/epoch_loss_Test', loss_epoch_test, n_iter)

        # Sortie directe (mean on epoch)
        if verbose:
            print(f"Itérations_train {n_iter}: loss {loss_epoch_train}")
            print(f"Itérations_test {n_iter}: loss {loss_epoch_test}")

if __name__=="__main__":
    # load, split and scale the data
    X, y = load_boston(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1,1), test_size=0.20, random_state=42) # ? random state ?
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

    # complete pass/ same result as question_1_2
    stochastic_gradient_descent(x_train_input, x_test_input, y_train_input, y_test_input, Model = Linear,\
                                Loss = MSE, nb_epochs = 2000, batch_size = len(x_train_input), learning_rate = 0.01,\
                                writer = SummaryWriter(), verbose = True)
    """

    # with batch passes, loss is calculated by mean after each epoch


    # mini batch passes
    stochastic_gradient_descent(x_train_input, x_test_input, y_train_input, y_test_input, Model = Linear,\
                                    Loss = MSE, nb_epochs = 1000, batch_size = 40, learning_rate = 0.0001,\
                                    writer = SummaryWriter(), verbose = True)
    """
