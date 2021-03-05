# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck
import pdb
import torch
from LINEAR_MSE import *
from torch.utils.tensorboard import SummaryWriter

# Description du contenu :
# Descente de gradient avec autograd sur des données jouet


if __name__ == "__main__":
    # Les données supervisées
    x = torch.randn(50, 13)
    y = torch.randn(50, 3)

    # Les paramètres du modèle à optimiser
    w = torch.randn((13, 3), requires_grad = True)
    b = torch.randn((3), requires_grad = True)


    epsilon = 0.05

    writer = SummaryWriter()
    for n_iter in range(100):
        # Calcul du forward 
        yhat = Linear.forward(x, w, b)
        loss = MSE.forward(y,yhat)
        loss.backward()

        # `loss` doit correspondre au coût MSE calculé à cette itération
        writer.add_scalar('Loss/train', loss, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")

        # Mise à jour des paramètres du modèle
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad

        w.grad.zero_()
        b.grad.zero_()
