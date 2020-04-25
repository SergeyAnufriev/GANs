
import numpy as np
import torch.nn as nn
from numpy.random import rand
from torch import device,cuda,optim,ones,zeros,tensor
from matplotlib import pyplot
import torch
from matplotlib.pyplot import plot as plt
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import wandb
import time

device2 = device("cuda:0" if cuda.is_available() else "cpu")

bsize = 64
n = 1000
z_dim = 1

activations  = {'LeakyReLU': nn.LeakyReLU(0.2, inplace=True),'Sigmoid': nn.Sigmoid(), 'Tanh': nn.Tanh()}

sweep_config = {'method':'grid','parameters':{'n_hidden_generator':{'values':[10,50,200]},\
                                              'n_hidden_discriminator':{'values':[10,50,200]}, \
                                              'activation':{'values':['LeakyReLU', 'Sigmoid', 'Tanh']}}}

config_defaults = {'n_hidden_generator': 10,'n_hidden_discriminator': 10,'activation':'LeakyReLU'}


sweep_id = wandb.sweep(sweep_config, entity="zcemg08", project="gans_training2")



def sample(n=bsize):
    # generate random inputs in [-0.5, 0.5]
    X1 = rand(n) - 0.5
    # generate outputs X^2 (quadratic)
    X2 = X1 * X1
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    return np.hstack((X1, X2))

datka = np.zeros((bsize*n,2))
for i in range(n):
    a = i*bsize
    b = (i+1)*bsize
    datka[a:b,:] = sample(bsize)

class Custom_dataset(Dataset):
    def __init__(self,datka):
        self.datka  = datka
    def __len__(self):
        return len(self.datka)
    def transform(self,x):
        return torch.from_numpy(x).float().to(device2)
    def __getitem__(self, idx):
        sample = self.datka[idx,:]
        sample = self.transform(sample)
        return sample


dataloader = DataLoader(Custom_dataset(datka), batch_size=64,shuffle=True,pin_memory=False)

class Discriminator(nn.Module):
    def __init__(self,activation_string,n_hidden_discriminator):
        super().__init__()
        self.D_lin1 = nn.Linear(2,n_hidden_discriminator)
        self.D_lin2 = nn.Linear(n_hidden_discriminator,1)
        self.act  = activations[activation_string]
        self.act2  = nn.Sigmoid()

    def forward(self, x):
        x = self.D_lin1(x)
        x = self.act(x)
        x = self.D_lin2(x)
        x = self.act2(x)
        return x
    
class Generator(nn.Module):
    def __init__(self,activation_string,n_hidden_generator):
        super().__init__()
        self.G_lin1 = nn.Linear(z_dim,n_hidden_generator)
        self.G_lin2 = nn.Linear(n_hidden_generator,2)
        self.act  = activations[activation_string]

    def forward(self,x):

        x = self.G_lin1(x)
        x = self.act(x)
        x = self.G_lin2(x)

        return x


def vis(G):
    with torch.no_grad():
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        data = sample()
        ax.scatter(data[:, 0], data[:, 1],c='r')
        G.eval()
        z_noise =torch.randn((100,z_dim),device=device2)
        gen = G(z_noise)
        gen = gen.cpu().data.numpy()
        ax.scatter(gen[:,0],gen[:,1],c='b')
        return fig

######### GAN Block receives param dict, remains unchanged #############
fixed_noise = torch.randn(bsize, 1, device=device2)


class GAN(pl.LightningModule):
    def __init__(self,activation_string,n_hidden_discriminator,n_hidden_generator):
        super(GAN, self).__init__()
        self.G = Generator(activation_string,n_hidden_generator).to(device2)
        self.D = Discriminator(activation_string,n_hidden_discriminator).to(device2)
        

    def forward(self, z):
        return self.G(z)

    def loss(self, y_hat, y):
        criterion = nn.BCELoss()
        return criterion(y_hat, y)
     
    def training_step(self, batch, batch_nb, optimizer_idx):
        x_real  = batch
        bsize   = batch.size(0)
        lab_real = ones(bsize, 1, device=device2)
        lab_fake = zeros(bsize, 1, device=device2)

        if optimizer_idx == 0:
            z = torch.randn(bsize, 1, device=device2)
            x_gen = self.G(z)
            D_G_z = self.D(x_gen)
            lossG = self.loss(D_G_z, lab_real)

            if batch_nb%100==0:
                wandb.log({'G_Loss':lossG.item(),"Generated vs Real ponints": vis(self.G)})
            
            return {'loss': lossG}

        if optimizer_idx == 1:

            D_x = self.D(x_real)
            lossD_real = self.loss(D_x, lab_real)
            z = torch.randn(bsize, 1, device=device2)
            x_gen = self.G(z).detach()
            D_G_z = self.D(x_gen)
            lossD_fake = self.loss(D_G_z, lab_fake)
            lossD = lossD_real + lossD_fake

            if batch_nb%100==0:
                wandb.log({'D_acc real': D_x.mean().item(),'D_acc fake':D_G_z.mean().item(),'D_Loss':lossD.item()})
            
            return {'loss': lossD}


    def configure_optimizers(self):
        
        opt_g = torch.optim.SGD(self.G.parameters(), lr=0.04)
        opt_d = torch.optim.SGD(self.D.parameters(), lr=0.04)

        return [opt_g, opt_d], [] ### Learning rate scheduler for now lr const
                      
    @pl.data_loader
    def train_dataloader(self):
        return dataloader


### CPU vs GPU, all set up in trainer pytorch light module ###
trainer = pl.Trainer(gpus=1,distributed_backend='dp', max_epochs = 10 ,bsize=64,progress_bar_refresh_rate=0,logger=False)


def train():
    wandb.init(config=config_defaults)
    config = wandb.config
    model = GAN(config.activation,config.n_hidden_discriminator,config.n_hidden_generator)
    wandb.watch(model,log='gradients')                  
    trainer.fit(model)


start = time.time()

wandb.agent(sweep_id, train)
                      
end = time.time()

print('Time to complete sweep = {}'.format(end - start))
