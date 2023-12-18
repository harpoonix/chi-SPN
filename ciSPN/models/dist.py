import torch
from torch.distributions import Distribution
from numpy.polynomial.hermite import hermgauss

# define the alpha-stable distribution in PyTorch
class AlphaStable(Distribution):
    def __init__(self, alpha, beta, mu, c):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.c = c
        super().__init__()

    def _cf(self, t):
        if (self.alpha == 1):
            return torch.exp(1j*t*self.mu - torch.pow(torch.abs(self.c*t), self.alpha)*(1 - 1j*self.beta*torch.sign(t)*(-2/torch.pi)*torch.log(torch.abs(t))))
        else:
            return torch.exp(1j*t*self.mu - torch.pow(torch.abs(self.c*t), self.alpha)*(1 - 1j*self.beta*torch.sign(t)*torch.tan(torch.pi*self.alpha/2)))
    
    def cf(self, t_vec, scope):
        t = t_vec[scope]
        return self._cf(t)
    
    def logpdf(self, x, scope):
        # use Gauss-Hermite quadrature to approximate the pdf
        t, w = hermgauss(50)
        f = lambda x, t: self._cf(t) * torch.exp(-1j * x * t) * torch.exp(t**2)
        I = torch.dot(w, f(x[scope], t))/(2*torch.tensor(torch.pi))
        return I.real