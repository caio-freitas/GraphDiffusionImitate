'''
Implementation of Variational Auto-Encoder Model
'''
from torch import nn


class VAE(nn.Module):
    def __init__(self,
                 input_dim, # output_dim = input_dim
                 hidden_dims=[64, 32, 16]
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                nn.ReLU()
                ) for i in range(len(hidden_dims)-1)
            ]
        )
        self.decoder = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(self.hidden_dims[-i], self.hidden_dims[-i-1]),
                nn.ReLU()
                ) for i in range(1, len(hidden_dims))
            ],
            nn.Linear(self.hidden_dims[0], input_dim),
            nn.Sigmoid()
        )
        self.latent_dim = self.hidden_dims[-1]  

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    