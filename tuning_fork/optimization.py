import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import itertools
import os
from PIL import Image as pil_img
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
def nice_print(*args):
    ret = []
    for k in args:
        if isinstance(k, (float, )):
            k = round(k, 10)
        ret.append(k)
    print(*ret); return

'''
Get grid of NxN intervals of the square of side 2s centered around the origin.
'''
def get_grid(s, N):
    x = torch.linspace(-s, s, N + 1)
    y = torch.linspace(-s, s, N + 1)
    x = x[: N] + s / N
    y = y[: N] + s / N
    grid_x, grid_y = torch.meshgrid(x, y)
    return torch.dstack((grid_x, grid_y)).reshape(-1, 2)


'''
Get all points that satisfy all line equations. Line equation can be either straight line or circle, according to line types. 
A line is defined by ax + by + c = 0. A shape is defined by ax + by + c < 0 for all lines defining it.
x - of shape (N, 2) - N 2-dimensional points.
eps - tolerance (default 0).
return - all points satisfying all equations, and boolean mask defining which points where chosen from the original.
'''
def filter_pts(x, lines, line_types, eps=0.):
    mask = torch.full((len(x),), True, dtype=torch.bool).to(DEVICE)
    for (a, b, c), line_type in zip(lines, line_types):
        if line_type == 'line':
            curr_mask = a * x[:, 0] + b * x[:, 1] + c < eps
        elif line_type == 'circle':
            center = torch.tensor([a, b])
            curr_mask = ((x.cpu() - center) ** 2).sum(dim=1) < c ** 2 + eps
        else:
            raise ValueError
        mask = mask & curr_mask.to(DEVICE)
    return x[mask], mask


'''
Get a random dense set of points on the boundary of a square, N + 1 points on each side.
s - half the side of the square.
'''
def get_square_boundary(s, N):
    range = torch.linspace(-s, s, N + 1).unsqueeze(-1)

    # add variation
    range1 = range + random_minus_one_one(range.shape) * s / (3 * N)
    range2 = range + random_minus_one_one(range.shape) * s / (3 * N)
    range3 = range + random_minus_one_one(range.shape) * s / (3 * N)
    range4 = range + random_minus_one_one(range.shape) * s / (3 * N)

    # range1 = s * random_minus_one_one((N + 1, 1))
    # range2 = s * random_minus_one_one((N + 1, 1))
    # range3 = s * random_minus_one_one((N + 1, 1))
    # range4 = s * random_minus_one_one((N + 1, 1))

    low = torch.full(range1.shape, -s)
    high = torch.full(range1.shape, s)
    left = torch.cat((low, range1), dim=-1)
    right = torch.cat((high, range2), dim=-1)
    up = torch.cat((range3, low), dim=-1)
    down = torch.cat((range4, high), dim=-1)
    return torch.cat((left, right, up, down), dim=0)

'''
Product of elements in list.
'''
def prod(lst):
    res = 1
    for elem in lst:
        res *= elem
    return res


def random_minus_one_one(shape):
    return 2 * (torch.rand(shape) - 0.5)

'''
Smooth approximation of the indicator function of a square.
a - half the side of the square.
t - temperature
'''
def square_indicator(a, t=1e10):
    return lambda x: torch.sigmoid(t * (x[:, 0] + a)) * torch.sigmoid(t * (a - x[:, 0])) * torch.sigmoid(t * (x[:, 1] + a)) * torch.sigmoid(t * (a - x[:, 1]))


'''
Smooth approximation of the indicator function of a line defined by the equation ax + by + c = 0 or a circle (x - a) ** 2 + (y - b) ** 2 = c ** 2.
t - temperature
eps - tolerance
'''
def line_indicator(a, b, c, t=1e5, eps=0., line_type='line'):
    if line_type == 'line':
        return lambda x: torch.sigmoid(-t * (a * x[:, 0] + b * x[:, 1] + c + eps))
    else:
        assert line_type == 'circle'
        center = torch.tensor([a, b]).to(DEVICE)
        return lambda x: torch.sigmoid(-t * (((x - center) ** 2).sum(dim=1) - (c ** 2 + eps)))

'''
Smooth approximation of the indicator function of the boundary of a general shape defined as an intersection of half-planes and circles.
lines - list of lines, each line defined by parameters a, b, c
line_types - list of strings - 'line' or 'circle'.
eps - paramter controlling boundary thickness.
boundary_val - value of the function on the boundary.
'''
def shape_indicator_fn(lines, line_types, eps=1e-4, boundary_val=200):
    indicators1 = [line_indicator(*line, line_type=line_type) for line, line_type in zip(lines, line_types)]
    indicators2 = [line_indicator(*line, line_type=line_type, eps=eps) for line, line_type in zip(lines, line_types)]

    return lambda x: boundary_val * prod([indicator(x) for indicator in indicators1]) * (1 - prod([indicator(x) for indicator in indicators2]))
    # return lambda x: 0.0007 * prod([indicator(x) for indicator in indicators1]) * (1 - prod([indicator(x) for indicator in indicators2]))
    # return lambda x: 0.005 * prod([indicator(x) for indicator in indicators1]) * (1 - prod([indicator(x) for indicator in indicators2]))
    # return lambda x: 0.01 * prod([indicator(x) for indicator in indicators1]) * (1 - prod([indicator(x) for indicator in indicators2]))

def draw_circle(image, r, scale=1):
    h, w = image.shape
    assert h == w
    x, y = np.arange(-h // 2, h // 2, dtype=int), np.arange(-h // 2, h // 2, dtype=int)
    xg, yg = np.meshgrid(x, y)
    rs = xg ** 2 + yg ** 2
    image[np.abs(rs - r ** 2) < 100 * scale] = 1

'''
Get line connecting two points.
'''
def get_line(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    # return [y2 - y1, -x2 + x1, -x1 * y2 + x2 * y1]
    return [-y2 + y1, x2 - x1, x1 * y2 - x2 * y1]

'''
Get unit length point corresponding to an angle.
'''
def angle2pt(angle):
    return (np.sin(angle), np.cos(angle))

'''
Compute parameter given other parameters according to the equation:
f = (N / (2pi * L ** 2)) * sqrt((E * I) / (rho * A))
'''


def compute_C(f, L, E, rho):
    N = 3.516015
    res = (((2 * np.pi * f * (L ** 2)) / N) ** 2) * rho / E
    return res


def compute_L(f, C, E, rho):
    N = 3.516015
    res = ((N / (2 * np.pi * f)) * ((C * E / rho) ** 0.5)) ** 0.5
    return res


def compute_f(L, E, rho, C):
    N = 3.516015
    return (N / (2 * np.pi * (L ** 2))) * ((E * C / rho) ** 0.5)


def set_lr(optim, lr):
    count = 0
    for g in optim.param_groups:
        old_lr = float(g['lr'])
        if old_lr > lr:
            count += 1
            g['lr'] = lr


def cut_excess_area(m):
    def _check(f, rng):
        for i in rng:
            if f(i):
                return i
    x = m
    t = 0
    sx = _check(lambda i: (x[i, :] > 0).float().mean() > t, range(x.shape[0]))
    ex = _check(lambda i: (x[i, :] > 0).float().mean() > t, reversed(range(x.shape[0])))
    sy = _check(lambda i: (x[:, i] > 0).float().mean() > t, range(x.shape[1]))
    ey = _check(lambda i: (x[:, i] > 0).float().mean() > t, reversed(range(x.shape[1])))
    return sx,ex,sy,ey


class SDF(nn.Module):
    '''
    Radial basis function with n elements, in the square [-a, a] X [-a, a], which is our domain
    If shape_function is not None, select only elements of RBF whose center is inside the shape
    If indicator_fn is not None, the SDF learns the residual w.r.t. the given indicator function.
    '''

    def __init__(self, n=100, a=0.005, indicator_fn=None,
                 basis_init='uniform', lambdas_init='uniform',
                 epsilon=3000,
                 epsilon_init='const', shape_function=None, verbose=False):
        super(SDF, self).__init__()
        self.verbose = verbose
        self.a = a
        self.n = n
        if basis_init == 'grid':
            basis = get_grid(self.a, int(self.n ** 0.5))
        elif basis_init == 'uniform':
            basis = self.a * random_minus_one_one((n, 2))
        elif basis_init == 'normal':
            basis = self.a * torch.randn((n, 2))
        if shape_function is not None:
            basis, _ = filter_pts(basis, *shape_function)

        self.n_basis = basis.shape[0]
        self.basis = nn.Parameter(basis)
        # self.basis = basis.to(DEVICE)

        self.shape_function = shape_function

        if lambdas_init == 'uniform':
            lambdas = 0.1 * self.a * random_minus_one_one((self.n_basis,))
        elif lambdas_init == 'normal':
            lambdas = self.a * torch.randn((self.n_basis,))
        self.lambdas = nn.Parameter(lambdas)

        if epsilon_init == 'const':
            epsilons = epsilon * torch.ones((self.n_basis,))
        elif epsilon_init == 'normal':
            epsilons = 0.2 * epsilon * torch.randn((self.n_basis,)) + epsilon
        self.epsilons = nn.Parameter(epsilons)
        # self.epsilons = epsilons.to(DEVICE)

        self.sigmoid = torch.nn.Sigmoid()
        self.indicator_fn = indicator_fn

    '''
    x of size N X 2 (N 2-dimensional points)
    '''
    def forward(self, x_inp):
        x = x_inp.unsqueeze(1).repeat((1, self.basis.shape[0], 1))
        basis = self.basis.unsqueeze(0).repeat((x.shape[0], 1, 1))
        sq_dist = ((x - basis) ** 2).sum(dim=-1)  # compute squared distance from each element of the basis, for each point
        exp = torch.exp(-(self.epsilons ** 2) * sq_dist)  # use exponent for the each radial function with factors epsilon
        res = (self.lambdas * exp).sum(dim=-1)  # weight the radial functions with lambdas
        if self.indicator_fn is not None: # In the case we learn the residual w.r.t. the indicator function
            ind = self.indicator_fn(x_inp)
            # print('indicator min max', ind.min(), ind.max())
            res += ind
        return res

    '''
    Integrate on the square [-a, a] ^ 2 differentiably with a sigmoid
    '''
    def integrate(self, N=256, t=1e0):
        pts = get_grid(self.a, N + 1).to(DEVICE)
        if self.shape_function is not None:
            pts, _ = filter_pts(pts, *self.shape_function)
        S = self.sigmoid(t * self(pts))
        # A = S.mean()
        # I = (S * (pts ** 2).sum(axis=-1)).mean()
        A = S.sum() / ((N + 1) ** 2)
        I = (S * (pts ** 2).sum(axis=-1)).sum() / ((N + 1) ** 2)
        return A, I, S

    '''
    Integrate on the square [-a, a] ^ 2 (not differentiable) for evaluation
    '''
    def integrate_eval(self, N=512, threshold=1e-8):
        with torch.no_grad():
            pts = get_grid(self.a, N + 1).to(DEVICE)
            if self.shape_function is not None:
                pts,_ = filter_pts(pts, *self.shape_function)
            S = (self(pts) > threshold).float()
            # A = S.mean()
            # I = (S * (pts ** 2).sum(axis=-1)).mean()
            A = S.sum() / ((N + 1) ** 2)
            I = (S * (pts ** 2).sum(axis=-1)).sum() / ((N + 1) ** 2)
        return A, I

    # '''
    # Run the SDF on a grid for evaluation
    # '''
    # def get_set(self, N=256, margin=1.5):
    #     int_margin = int(margin * (N + 1))
    #     pts = get_grid(margin * self.a, int_margin).to(DEVICE)
    #     S = self(pts)
    #     if self. shape_function is not None:
    #         _, mask = filter_pts(pts, *self.shape_function)
    #         S[~mask] = 0
    #     S = S.reshape((int_margin, int_margin))
    #     S = S.detach().cpu().numpy()
    #     return S
    '''
    Run the SDF on a grid for evaluation
    '''
    def get_set(self, N=256, margin=1.5, cut_to_shape=False, ret_mask=False):
        int_margin = int(margin * (N + 1))
        pts = get_grid(margin * self.a, int_margin).to(DEVICE)
        S = self(pts)
        mask = None
        if self.shape_function is not None:
            _, mask = filter_pts(pts, *self.shape_function)
            S[~mask] = 0
            mask = mask.reshape((int_margin, int_margin))
        S = S.reshape((int_margin, int_margin))
        
        if cut_to_shape:
            if self.shape_function is not None:
                sx,ex,sy,ey = cut_excess_area(mask)
                mask = mask[sx:ex+1,sy:ey+1]
            else:
                sx,ex,sy,ey = cut_excess_area(S)
            S = S[sx:ex+1,sy:ey+1]

        S = S.detach().cpu().numpy()
        if ret_mask:
            mask = mask.detach().cpu().numpy()
            return S, mask
        return S

    def repulsion_loss(self, epsilon=0.001):
        expanded1 = self.basis.unsqueeze(1).repeat((1, self.n_basis, 1))
        expanded2 = self.basis.unsqueeze(0).repeat((self.n_basis, 1, 1))
        dist = (torch.abs(expanded1 - expanded2)).sum(dim=-1)  # compute distance for each pair of the basis
        epsilon = torch.full(dist.shape, epsilon * self.a).to(DEVICE)
        zero = torch.zeros_like(dist).to(DEVICE)
        return F.l1_loss(torch.maximum(epsilon - dist, zero), zero, reduction='sum')

    def weight_decay(self):
        return F.l1_loss(torch.abs(self.lambdas), torch.zeros_like(self.lambdas), reduction='sum')

    def boundary_loss(self, N=256, eps=0.01):
        pts = get_grid(self.a, N + 1).to(DEVICE)

        boundary = get_square_boundary(self.a, N + 1).to(DEVICE)
        boundary_values = self(boundary)
        zero = torch.zeros_like(boundary_values).to(DEVICE)

        boundary2 = get_square_boundary((1 + eps) * self.a, N + 1).to(DEVICE)
        boundary_values2 = self(boundary2)

        term1 = F.l1_loss(torch.minimum(boundary_values, zero), zero, reduction='sum')
        term2 = F.l1_loss(torch.maximum(boundary_values2, zero), zero, reduction='sum')
        return term1 + term2

    '''
    Compute loss for equation I / A = C, in log scale. 
    A_weight - regularization term for A area
    N - grid size for integration
    '''

    def compute_loss(self, C, N=256, eps=1e-15,
                     A_weight=0.1, entropy_weight=1e-4, repulsion_weight=1., weight_decay=0.01,
                     rep_epsilon=1e-3, sigmoid_t=1e0, boundary_weight=0.01):
        A, I, S = self.integrate(N=N, t=sigmoid_t)
        C = torch.tensor(C).to(DEVICE)
        loss_f = F.l1_loss
        C_loss = loss_f(torch.log(I + eps) - torch.log(A + eps), torch.log(C + eps))
        A_loss = loss_f(torch.log(A + eps), torch.log(torch.zeros_like(A) + eps))
        entropy_loss = loss_f(S * (1 - S), torch.zeros_like(S).to(DEVICE))
        repulsion_loss = self.repulsion_loss(epsilon=rep_epsilon)
        weight_decay_loss = self.weight_decay()
        boundary_loss = self.boundary_loss()
        if self.verbose:
            nice_print('losses', C_loss.item(), 'A, I', A.item(), I.item(), 'I/A', (I / A).item(), 'C', C.item(), 'logs', np.log((I / A).item()), np.log(C.item()))
            nice_print('entropy', entropy_loss.item(), 'repulsion', repulsion_loss.item(), 'weight decay', weight_decay_loss.item(), 'boundary', boundary_loss.item())
        A_eval, I_eval = self.integrate_eval(N=N)
        C_eval = (I_eval / A_eval).item()
        C_continuous = (I / A).item()
        if self.verbose:
            nice_print('Actual I / A:', np.log(C_eval), 'C', np.log(C.item()))
        combined_loss = C_loss + A_weight * A_loss + entropy_weight * entropy_loss + repulsion_weight * repulsion_loss + weight_decay * weight_decay_loss + boundary_weight * boundary_loss
        return combined_loss, C_eval, C_continuous



