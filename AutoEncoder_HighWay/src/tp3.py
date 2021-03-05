from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import pdb
import torch.optim as optim


# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()
#pdb.set_trace()
# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
#writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
#images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
#images = make_grid(images)
# Affichage avec tensorboard
#writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")

#  TODO:

# définiton du dataset
class MonDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __getitem__(self, index):
        image = torch.tensor(self.data[index].reshape(-1), dtype=torch.float32)
        return (image, torch.tensor([self.labels[index]]))
    def __len__(self):
        return len(self.data)# a verifier si la class data adment les len ou il faut utiliser shape

class AutoEncoder(nn.Module):
    def __init__(self, inputShape, ):
        super(AutoEncoder, self).__init__()
        self.inputShape = inputShape
        self.encoder1 = nn.Linear(inputShape, 128)
        self.decoder2 = nn.Linear(128, inputShape)
        #torch.matmul(m.weight.t(), k)
        #self.decoder1 = nn.Linear(self.encoder1.weight.t())

    def forward(self, x):
        out_1 = self.encoder1(x)
        # tied autoencoder
        out_2 = F.linear(F.relu(out_1),self.encoder1.weight.t())
        return F.sigmoid(out_2)
if __name__ == "__main__":
    # dataSet
    data = MonDataset(train_images, train_labels)
    # model
    model = AutoEncoder(28*28)

    # Hyperparameters
    num_epochs = 1
    batch_size = 4
    lr = 0.002
    optimizer = optim.SGD(model.parameters(), lr)
    # data loader
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    criterion = nn.MSELoss()



    for epoch in range(num_epochs):
        print("Epoch %d" % epoch)
        totloss = 0
        for i, (image, _) in enumerate(data_loader):

            out = model(image)

            optimizer.zero_grad()
            loss = criterion(out, image)
            loss.backward()
            totloss += float(loss)
            optimizer.step()
            if (i+1 )% 100 == 0:
              print(i)
              print("mean :", totloss / i)
            del image
