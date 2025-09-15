import torch
from multi_elo_rating import get_thurstonian_rating_gradient, gaussian_log_density

mu = torch.tensor([0.0, 1.0, 2.0, 3.0])
n = mu.shape[0]
r = torch.full([n], torch.mean(mu))
num_simulations = 1000
print(torch.mean(mu))

lr = 0.005
# beta1 = 0.9
# beta2 = 0.999
# eps = 1e-15
# g1 = torch.zeros_like(r)
# g2 = torch.zeros_like(r)

for i in range(num_simulations):
    X = torch.randn([n]) + mu
    sorted_x, indices = torch.sort(-X)
    ranks = torch.argsort(indices)
    g = torch.tensor(get_thurstonian_rating_gradient(r, ranks, gaussian_log_density, 5.0))
    
    # g1 = g1 * beta1 + (1 - beta1) * g
    # g2 = g2 * beta2 + (1 - beta2) * torch.square(g)
    # step = lr * g1 / torch.sqrt(g2 + eps)
    # r = r + step
    r = r + lr * g
    print(r, torch.mean(r))