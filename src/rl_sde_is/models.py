import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity()):

    # preallocate layers list
    layers = []

    for j in range(len(sizes)-1):

        # actiavtion function
        act = activation if j < len(sizes)-2 else output_activation

        # linear layer
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]

    # Sequential model with given layers
    return nn.Sequential(*layers)

class FeedForwardNN(nn.Module):
    def __init__(self, d_in, hidden_sizes, d_out, activation=nn.Tanh(),
                 output_activation=nn.Identity(), seed=None):
        super(FeedForwardNN, self).__init__()

        # set seed
        if seed is not None:
            torch.manual_seed(seed)

        # hidden_sizes
        self.d_in = d_in
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1
        self.d_out = d_out
        self.d_layers = [d_in] + hidden_sizes + [d_out]

        # define linear layers
        for i in range(self.n_layers):
            setattr(
                self,
                'linear{:d}'.format(i+1),
                nn.Linear(self.d_layers[i], self.d_layers[i+1], bias=True),
            )

        # activation function
        self.activation = activation
        self.output_activation = output_activation

    def forward(self, x):
        for i in range(self.n_layers):

            # linear layer
            layer = getattr(self, 'linear{:d}'.format(i+1))

            # forward layer pass with chosen layer activation
            if i != self.n_layers -1:
                x = self.activation(layer(x))

            # last forward layer pass with output activation
            else:
                x = self.output_activation(layer(x))
        return x

class DenseNN(nn.Module):
    def __init__(self, d_in, hidden_sizes, d_out, activation_type=torch.tanh, seed=None):
        super(DenseNN, self).__init__()

        # set seed
        if seed is not None:
            torch.manual_seed(seed)

        # layer dimensions
        self.d_in = d_in
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1
        self.d_out = d_out
        self.d_layers = [d_in] + hidden_sizes + [d_out]


        # define linear layers
        for i in range(self.n_layers):
            setattr(
                self,
                'linear{:d}'.format(i+1),
                nn.Linear(int(np.sum(self.d_layers[:i+1])), self.d_layers[i+1], bias=True),
            )

        # activation function
        self.activation = activation

    def forward(self, x):
        for i in range(self.n_layers):

            # linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))

            # forward layer pass with activation
            if i != self.n_layers - 1:
                x = torch.cat([x, self.activation(linear(x))], dim=1)

            # last forward layer pass without activation
            else:
                x = linear(x)
        return x
