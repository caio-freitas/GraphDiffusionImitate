from torch import nn

class MLPNet(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_dims=[64, 64],
                 activation=nn.ReLU(),
                 output_activation=nn.Identity(),
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_activation = output_activation
        
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            self.activation,
            *[nn.Sequential(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                self.activation,
            ) for i in range(len(self.hidden_dims)-1)],
            nn.Linear(self.hidden_dims[-1], self.output_dim)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        x = self.output_activation(x)
        return x