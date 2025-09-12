import torch
import spline

def log_density(rating, x):
    return -torch.square(x - rating)

def get_rating_log_prob(ratings, num_censored=0):
    a = min(ratings).detach() - 5
    b = max(ratings).detach() + 5
    k = 10000   # number of spline nodes to use for the numerical integration
    sp = spline.Spline(k, a, b, dtype=torch.double)
    x = sp.x

    # Rankings tied for last place would be from DNF/DQ, where the racer
    # quit before completing the race. These are treated as censored,
    # where all we assume is that they would have finished after all others.
    i = 0
    F = 0
    while i < num_censored:
        F = F + sp.log_antideriv(log_density(ratings[i], x))
        i += 1

    while i < len(ratings):
        f = F + log_density(ratings[i], x)
        F = sp.log_antideriv(f)
        i += 1

    log_prob = F[-1]
    return log_prob

def get_rating_gradient(ratings, ranks):
    ratings = torch.tensor(ratings, dtype=torch.double, requires_grad=True)
    ranks = torch.tensor(ranks, dtype=torch.int64)
    sorted_ranks, indices = torch.sort(ranks, descending=True)
    num_censored = torch.sum(sorted_ranks == sorted_ranks[0])
    ordered_ratings = ratings[indices]
    log_prob = get_rating_log_prob(ordered_ratings, num_censored)
    log_prob.backward()
    return ratings.grad.tolist()
  