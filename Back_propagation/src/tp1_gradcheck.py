import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)

print(torch.autograd.gradcheck(mse, (yhat, y)))

#  TODO:  Test du gradient de Linear
q = 10 # le nombre d'exemples
n = 6 # le nombre de features
p = 5 # le nombre de classes i.e le nombre de W (les w sont en vecteurs colonnes)
X = torch.randn(q, n, requires_grad=True, dtype=torch.float64)
W = torch.randn(n, p, requires_grad=True, dtype=torch.float64)
b = torch.randn(1, p, requires_grad=True, dtype=torch.float64)

print(torch.autograd.gradcheck(linear, (X, W, b)))
