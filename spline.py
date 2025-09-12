import torch
import math

# Cubic splines for numerical integration

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
            torch.full_like(log_y1[0:1], float('-inf')),
            log_y1
        ], dim=0)
        return torch.logcumsumexp(out, dim=0) + math.log(self.scale)
