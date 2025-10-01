import torch
import numpy as np
import math


def pairwise_elo_gradient(ratings, ranks, use_average=bool):
    out = []
    for i, r1 in enumerate(ratings):
        g = 0
        cnt = 0
        for j, r2 in enumerate(ratings):
            if ranks[i] < ranks[j]:
                g += 1.0 / (1.0 + math.exp(r1 - r2))
                cnt += 1
            elif ranks[i] > ranks[j]:
                g -= 1.0 / (1.0 + math.exp(r2 - r1))
                cnt += 1
        if use_average and cnt > 0:
            g /= cnt
        out.append(g)
    return out


def plackett_luce_gradient(ratings: list[float], inverted: bool) -> list[float]:
    """
    The ratings are assumed to be given in order by outcome, with the winner's rating listed first.
    Ties are not supported.
    """
    ratings = torch.tensor(ratings, dtype=torch.double)
    if inverted:
        ratings = -ratings
    else:
        ratings = torch.flip(ratings, [0])        
    lam = torch.exp(-ratings)
    clam = torch.flip(torch.cumsum(torch.flip(lam, [0]), 0), [0])
    inv_clam = torch.cumsum(1.0 / clam, 0)
    out = -1.0 + lam * inv_clam
    if inverted:
        out = -out
    else:
        out = torch.flip(out, [0])
    return out.tolist()


def plackett_luce_gradient_with_ranks(ratings: list[float], ranks: list[float]) -> list[float]:
    ratings = torch.tensor(ratings, dtype=torch.double, requires_grad=True)
    ranks = torch.tensor(ranks, dtype=torch.int64)
    sorted_ranks, indices = torch.sort(ranks, descending=True)
    num_censored = torch.sum(sorted_ranks == sorted_ranks[0])
    assert num_censored < len(ratings)
    ordered_ratings = ratings[indices]
    
    clam0 = torch.flip(torch.logcumsumexp(torch.flip(-ordered_ratings, [0]), 0), [0])
    a = clam0[num_censored]
    clam = torch.cat([torch.logaddexp(-ordered_ratings[:num_censored], a), clam0[num_censored:]], 0)
    L = torch.sum(-ordered_ratings) - torch.sum(clam)
    L.backward()
    return ratings.grad.tolist()

class Spline:
    def __init__(self, n, a, b, dtype=torch.double):
        super().__init__()
        self.n = n
        self.a = a
        self.b = b
        self.scale = (b - a) / (n - 1)
        self.x = torch.arange(n, dtype=dtype) * self.scale + a

    def antideriv(self, y):
        y_pad = torch.cat([
            torch.zeros_like(y[0:1]),
            y,
            torch.zeros_like(y[0:1]),
        ], dim=0)
        out = torch.cat([
            torch.zeros_like(y[0:1]),
            -1/24 * y_pad[:-3] + 13/24 * y_pad[1:-2] + 13/24 * y_pad[2:-1] - 1/24 * y_pad[3:]
        ])
        return torch.cumsum(out, dim=0) * self.scale

    def log_antideriv(self, log_y):
        log_y_pad = torch.cat([
            torch.full_like(log_y[0:1], float('-inf')),
            log_y,
            torch.full_like(log_y[0:1], float('-inf')),
        ], dim=0)
        A = torch.stack([log_y_pad[:-3], log_y_pad[1:-2], log_y_pad[2:-1], log_y_pad[3:]], dim=0)
        c = torch.amax(A, dim=0, keepdim=True)
        A = A - c
        eA = torch.exp(A)
        y1 = -1/24 * eA[0] + 13/24 * eA[1] + 13/24 * eA[2] - 1/24 * eA[3]
        y1 = torch.clamp_min(y1, 1e-30)
        log_y1 = torch.log(y1) + c[0]
        out = torch.cat([
            torch.full_like(log_y1[0:1], -1e30),  # avoid -inf since it results in NaN gradients
            log_y1
        ], dim=0)
        return torch.logcumsumexp(out, dim=0) + math.log(self.scale)


def get_rating_log_prob(ratings, weights, num_censored, log_density_fn, margin):
    # the +/-margin is to approximately cover the range on which the density functions
    # are non-negligibly different from zero:
    a = min(ratings).detach() - margin
    b = max(ratings).detach() + margin
    k = 500   # number of spline nodes to use for the numerical integration
    sp = Spline(k, a, b, dtype=torch.double)
    x = sp.x

    # Rankings tied for last place would be from DNF/DQ, where the player
    # quit before completing the game. These are treated as censored,
    # where all we assume is that they would have finished after all others.
    i = 0
    F = 0
    while i < num_censored:
        F = F + weights[i] * sp.log_antideriv(log_density_fn(x - ratings[i]))
        i += 1

    while i < len(ratings):
        f = F + log_density_fn(x - ratings[i])
        F = sp.log_antideriv(f)
        i += 1

    log_prob = F[-1]
    return log_prob


def gaussian_log_density(x):
    return -torch.square(x) / 2


def skew_gaussian_log_density(alpha):
    def f(x):
        return -torch.square(x) / 2 + torch.special.log_ndtr(alpha * x)
    return f


def logerfc(x): 
    return torch.where(x > 0.0,
                       torch.log(torch.special.erfcx(x)) - x**2,
                       torch.log(torch.special.erfc(x)))

def exp_modified_gaussian_log_density(lam):
    def f(x):
        return lam * x + logerfc((lam + x) / math.sqrt(2))
    return f


def hyperbolic_secant_density(x):
    return -torch.logaddexp(x, -x)


def skew_hyperbolic_secant_density(alpha):
    def f(x):
        return -torch.logaddexp(x * alpha, -x)
    return f


def hyperbolic_exp_density(x):    
    return -torch.sqrt(torch.square(x) + 1)


def get_thurstonian_rating_gradient(ratings: list[float], ranks: list[int], weights: list[float], log_density_fn, margin: float) -> list[float]:
    """
    Given a list of player ratings and a list of corresponding ranks on a given game,
    return a list of gradients of the log-likelihood with respect to those ratings.
    This indicates the direction that the ratings should be updated in order to
    increase the likelihood of the game's outcome. 
    :param ratings: A list of player ratings.
    :param ranks: A corresponding list of player ranks in the outcome of the game. 
      Ties are supported only for last place (i.e. highest-numbered rank).
    :param weights: Only supported for last place (i.e. highest-numbered rank).
    :param log_density_fn: Log of the density function to be used.
    :param margin: Value such that [-margin, margin] approximately captures the support of the density function.
    """
    ratings = torch.tensor(ratings, dtype=torch.double, requires_grad=True)
    weights = torch.tensor(weights, dtype=torch.double)
    ranks = torch.tensor(ranks, dtype=torch.int64)
    sorted_ranks, indices = torch.sort(ranks, descending=True)
    num_censored = torch.sum(sorted_ranks == sorted_ranks[0])
    ordered_ratings = ratings[indices]
    ordered_weights = weights[indices]
    log_prob = get_rating_log_prob(ordered_ratings, ordered_weights, num_censored, log_density_fn, margin)
    log_prob.backward()
    return ratings.grad.tolist()
  
  

def get_updated_ratings_simple(
    ratings: list[float],
    ratings_grad: list[float],
    learning_rate: float,
) -> list[float]:
    """
    Given a list of player ratings and a list of corresponding ranks on a given game,
    return a list of updated player ratings.
    :param ratings: A list of player ratings.
    :param ratings_grad: A list of player rating gradient values.
    :param learning_rate: the learning rate (a.k.a. step size).
    :return: A list of updated player ratings.
     """
    out = []
    for i in range(len(ratings)):
        old_rating = ratings[i]
        grad = ratings_grad[i]
        new_rating = old_rating + learning_rate * grad
        out.append(new_rating)
    return out

def get_updated_ratings(
    ratings: list[float],
    ratings_grad: list[float],
    rating_floor: float,
    knots: list[float],
    learning_rates: list[float],
) -> list[float]:
    """
    Given a list of player ratings and a list of corresponding ranks on a given game,
    return a list of updated player ratings.
    :param ratings: A list of player ratings.
    :param ratings_grad: A list of player rating gradient values.
    :param rating_floor: the minimum possible rating; any rating that drops below this will be clamped to this value.
    :param learning_rate_base: the maximum learning rate (a.k.a. step size), applied for ratings at the rating floor.
    :param learning_rate_decay: the rate at which the learning rate decays exponentially as a function of the rating.
    :param negative_scaling_base: the maximum factor subtracted from negative rating gradients, to scale them down.
      This is a way of softening the rating floor, to make it more continuous. It also functions as a sort of activity 
      bonus for new players. This effect is fully applied at the rating floor and then tapers off exponentially. 
      A value of 0 won't apply any scaling to negative gradients. A value of 1 will completely zero out negative gradients
      at the rating floor.
    :param negative_scaling_rate: the rate of exponential decay of negative gradient scaling.
    :return: A list of updated player ratings.
     """
    out = []
    for i in range(len(ratings)):
        old_rating = ratings[i]
        grad = ratings_grad[i]
        lr = np.interp(old_rating, knots, learning_rates)
        new_rating = old_rating + lr * grad
        if new_rating < rating_floor:
            new_rating = rating_floor
        out.append(new_rating)
    return out