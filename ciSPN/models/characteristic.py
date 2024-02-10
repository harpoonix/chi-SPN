import torch

def ECF(t, data_x, individual = False):
    """Empirical Characteristic Function for the data
    Args:
        t: vector of size (D, N) where D is dimensionality, N is number of samples of t
        data_x: data of size (B, D) where B is batch size, D is dimensionality
        
    Returns:
        vector of size (1, N) where N is number of samples of t"""
    if individual:
        return torch.exp(1j * torch.matmul(data_x, t))
    return torch.mean(torch.exp(1j * torch.matmul(data_x, t)), dim=0, keepdim=True)

# def CFD(cc : CiSPN, x, sigma = 1, d = 4, n = 16):
#     """Characteristic Function Distance between the estimated CF from the model,
#     and the empirical CF from the data.
#     Args:
#         cc: CiSPN model
#         x: data of size (B, D) where B is batch size, D is dimensionality"""
    
#     t = torch.randn(d, n).to(x.device)
#     cf_vec = torch.cat([cc.root.cf(t[:, i].reshape(1, -1)*sigma) for i in range(n)])
#     return torch.mean(torch.square(torch.abs(cf_vec - ECF(t, x))))