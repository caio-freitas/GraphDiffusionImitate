'''
Implementation of Variational Auto-Encoder Model
'''
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self,
                 input_dim, # output_dim = input_dim
                 hidden_dims=[64, 32, 16],
                 activation=nn.ReLU(),
                 output_activation=nn.Identity(),
                 latent_dim=2
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                activation
                ) for i in range(len(hidden_dims)-1)
            ]
        )
        self.latent_dim =  latent_dim
        # latent mean and variance 
        self.mean_layer = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.logvar_layer = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[-1]),
            *[nn.Sequential(
                nn.Linear(self.hidden_dims[-i], self.hidden_dims[-i-1]),
                activation
                ) for i in range(1, len(hidden_dims))
            ],
            nn.Linear(self.hidden_dims[0], input_dim),
            output_activation
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  
        z = mean + var*epsilon
        return z

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5*logvar))
        x_hat = self.decode(z)
        return x_hat, mean, logvar


    def decode(self, x):
        return self.decoder(x)
    