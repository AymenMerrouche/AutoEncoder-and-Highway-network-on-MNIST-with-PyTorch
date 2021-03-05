# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck
import pdb

# doc : https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        #  TODO:  Renvoyer la valeur de la fonction
        q = yhat.shape[0]
        return (1/q) * (torch.norm(yhat-y)**2)
    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        q = yhat.shape[0]
        grad_mse_y = (-2/q) * grad_output * (yhat - y)
        grad_mse_yhat = -grad_mse_y
        return grad_mse_yhat, grad_mse_y


mse = MSE.apply


#  TODO:  Implémenter la fonction Linear(X, W, b)
class Linear(Function):
    """Début d'implementation de la fonction Linear"""
    @staticmethod
    def forward(ctx, x, w, b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(x, w, b)

        #  TODO:  Renvoyer la valeur de la fonction
        return x @ w + b
    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        x, w, b = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)

        grad_f_w = torch.transpose(x, 0, 1) @ grad_output
        grad_f_x = grad_output @ torch.transpose(w, 0, 1)
        grad_f_b = torch.sum(grad_output, dim=0).reshape(1, -1)
        return grad_f_x, grad_f_w, grad_f_b

linear = Linear.apply
