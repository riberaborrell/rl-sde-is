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

class GaussianAnsatzModel(nn.Module):

    def __init__(self, sde, m_i=20, sigma_i=0.5, normalized=True, seed=None):
        super(GaussianAnsatzModel, self).__init__()

        # normalize Gaussian flag
        self.normalized = normalized

        # set seed
        if seed is not None:
            torch.manual_seed(seed)

        # dimension and inverse of temperature
        self.d = sde.d

        # covariance matrix
        self.sigma_i = sigma_i
        self.cov = sigma_i * torch.eye(self.d)

        # compute inverse
        self.inv_cov = torch.eye(self.d) / sigma_i

        # compute determinant
        self.det_cov = sigma_i**self.d

        # set number of gaussians
        self.m_i = m_i
        self.m = m_i ** self.d
        self.d_flat = self.m

        # distribute centers of Gaussians uniformly
        """
        mgrid_input = []
        for i in range(self.d):
            slice_i = slice(self.sde.domain[i, 0], self.sde.domain[i, 1], complex(0, m_i))
            mgrid_input.append(slice_i)
        means = np.mgrid[mgrid_input]
        means = np.moveaxis(means, 0, -1).reshape(self.m, self.d)
        """
        self.means = torch.linspace(sde.state_space_bounds[0], sde.state_space_bounds[0], self.m_i).unsqueeze(dim=1)
        #self.means = torch.tensor(means, dtype=torch.float32)

        # set parameters
        self.theta = torch.nn.Parameter(torch.randn(self.m))


    def mvn_pdf_basis(self, x):
        # assume shape of x array to be (K, d)
        assert x.ndim == 2, ''
        assert x.size(1) == self.d, ''
        K = x.size(0)

        # compute log of the basis

        log_mvn_pdf_basis = - 0.5 * torch.sum(
            (x.view(K, 1, self.d) - self.means.view(1, self.m, self.d))**2,
            axis=2,
        ) / self.sigma_i

        # add normalization factor
        if self.normalized:
            log_mvn_pdf_basis -= torch.log(2 * torch.tensor(torch.pi) * self.sigma_i) \
                               * self.d / 2

        return torch.exp(log_mvn_pdf_basis)

    def mvn_pdf_gradient_basis(self, x):
        # assume shape of x array to be (K, d)
        assert x.ndim == 2, ''
        assert x.size(1) == self.d, ''
        K = x.size(0)

        # get nd gaussian basis
        mvn_pdf_basis = self.mvn_pdf_basis(x)

        # compute gradient of the exponential term
        grad_exp_term = (
            x.view(K, 1, self.d) - self.means.view(1, self.m, self.d)
        ) / self.sigma_i

        # compute gaussian gradients basis
        return - grad_exp_term * mvn_pdf_basis.unsqueeze(dim=2)

    def forward(self, x):
        x = self.mvn_pdf_gradient_basis(x)
        x = torch.tensordot(x, self.theta, dims=([1], [0]))
        return x

    def get_parameters(self):
        ''' get parameters of the model
        '''
        return self._parameters['theta'].detach().numpy()

    def load_parameters(self, theta):
        ''' load model parameters.
        '''
        assert theta.ndim == 1, ''
        assert theta.shape[0] == self.d_flat, ''
        self._parameters['theta'] = torch.tensor(
            theta,
            requires_grad=True,
            dtype=torch.float,
        )

    def get_rel_path(self):
        if self.distributed == 'uniform':
            rel_path = os.path.join(
                'gaussian-ansatz-nn',
                'm_{}'.format(self.m),
            )
        elif self.distributed == 'meta':
            rel_path = os.path.join(
                'gaussian-ansatz-nn',
                'meta-distributed',
            )
        return rel_path

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


def flatten_params(parameters):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**
    """
    l = [torch.flatten(p) for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    return {"params": flat, "indices": indices}


def recover_flattened(flat_params, indices, model):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    l = [flat_params[s:e] for (s, e) in indices]
    for i, p in enumerate(model.parameters()):
        l[i] = l[i].view(*p.shape)
