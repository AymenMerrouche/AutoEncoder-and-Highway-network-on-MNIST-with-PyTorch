import torch
from torch.autograd import Function
import torch
import numpy as np


# description du contenu
# Les modules MSE et linear



class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        q = yhat.shape[0]
        return (1/q) * (torch.norm(yhat-y)**2)

mse = MSE.apply


class Linear(Function):
    """Début d'implementation de la fonction Linear"""
    @staticmethod
    def forward( x, w, b):
        ## Garde les valeurs nécessaires pour le backwards
        return x @ w + b

linear = Linear.apply
