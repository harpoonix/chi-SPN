import torch
from torch.distributions import Distribution
from numpy.polynomial.hermite import hermgauss
from scipy.stats import levy_stable

# define the alpha-stable distribution in PyTorch
class AlphaStable(Distribution):
    arg_constraints = {}
    def __init__(self, alpha, beta, mu, c, n):
        """Give the optimal parameters for the AlphaStable Distribution
        
        Args:
            alpha (float): stability parameter
            beta (float): skewness parameter
            mu (float): location parameter
            c (float): scale parameter
            n (int) : number of distributions
        
        Formulae to compute actual parameters:
            alpha = 2/(1 + exp(-alpha)) \\
            beta = 2/(1 + exp(-beta)) - 1 \\
            mu = mu \\
            c = exp(c) \\
            """
        self.alpha = (2)/(1 + torch.exp(-alpha)).reshape(-1,n) + 1e-8
        self.beta = (2/(1 + torch.exp(-beta)) - 1).reshape(-1,n)
        self.mu = mu.reshape(-1,n)
        self.c = torch.exp(c).reshape(-1,n) + 0.01
        self.n = n
        super().__init__()

    def _cf(self, t):
        stable_cf = torch.zeros_like(self.alpha)
        assert(torch.all(self.c != 0) and torch.all(self.alpha != 0))
        
        if (self.alpha == 1).any():
            mask = self.alpha == 1
            # convert boolean tensor mask to integer
            mask = mask.type(torch.int)
            alpha_one = torch.exp(1j*t*self.mu - torch.pow(torch.abs(self.c*t), self.alpha)*(1 - 1j*self.beta*torch.sgn(t)*(-2/torch.pi)*torch.log(torch.abs(t))))
            alpha_other = torch.exp(1j*t*self.mu - torch.pow(torch.abs(self.c*t), self.alpha)*(1 - 1j*self.beta*torch.sgn(t)*torch.tan(torch.pi*self.alpha*(1-mask)/2)))
            stable_cf = mask*alpha_one + (1 - mask)*alpha_other
        else:
            stable_cf = torch.exp(1j*t*self.mu - torch.pow(torch.abs(self.c*t), self.alpha)*(1 - 1j*self.beta*torch.sgn(t)*torch.tan(torch.pi*self.alpha/2)))
        return stable_cf.reshape(-1, self.n)
    
    def cf(self, t_vec, scope):
        t = t_vec[scope]
        return self._cf(t)
    
    def log_prob(self, x):
        # use Gauss-Hermite quadrature to approximate the pdf
        t, w = hermgauss(50)
        w = torch.tensor(w, dtype = torch.complex64).reshape(-1, 1).to(x.device)
        t = torch.tensor(t, dtype = torch.complex64).reshape(-1, 1).to(x.device)
        """
        w = 50 weights
        t = 50 sample points
        """
        f = lambda x, t_vec: torch.stack([self._cf(t) * torch.exp(-1j * x * t) * torch.exp(t**2) for t in t_vec], dim = -1)
        I = torch.matmul(f(x, t), w)/(2*torch.tensor(torch.pi))
        logprob = torch.log(I.real).squeeze(-1)
        logprob = logprob.nan_to_num(nan = -1e8, posinf = -1e8, neginf = -1e8)
        return logprob
    
    def expensive_log_prob(self, x):
        external_dist = levy_stable(alpha = self.alpha.cpu().numpy(), beta = self.beta.cpu().numpy(), loc = self.mu.cpu().numpy(), scale = self.c.cpu().numpy())
        logprob = torch.tensor(external_dist.logpdf(x.cpu().numpy())).to(x.device)
        torch.nan_to_num(logprob, nan = -1e8, posinf = -1e8, neginf = -1e8, out=logprob)
        try:
            assert(torch.all(torch.isfinite(logprob)))
            assert(logprob.shape == x.shape)
        except AssertionError as e:
            print(logprob)
            print(e)
        return logprob.reshape(-1, 1)