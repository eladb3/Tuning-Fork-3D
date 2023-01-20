import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import itertools
import os
from PIL import Image as pil_img

'''
Get grid of NxN intervals of the square of side 2s centered around the origin.
'''

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def var2str(x):
    ROUND=10
    if isinstance(x, (float, int)):
        ret = round(x,ROUND)
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        ret = round(x.item(), ROUND)
    elif isinstance(x, (dict)):
        ret = {k:var2str(v) for k,v in x.items()}
    else:
        ret = x
    return ret

def nice_print(*args):
    print(*args); return
    # print(" ".join([str(var2str(x)) for x in args]))

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






def get_square_boundary(s, N):
    range = torch.linspace(-s, s, N + 1).unsqueeze(-1)

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

def prod(lst):
    res = 1
    for elem in lst:
        res *= elem
    return res

def random_minus_one_one(shape):
    return 2 * (torch.rand(shape) - 0.5)


def square_indicator(a, t=1e10):
    return lambda x: torch.sigmoid(t * (x[:, 0] + a)) * torch.sigmoid(t * (a - x[:, 0])) * torch.sigmoid(t * (x[:, 1] + a)) * torch.sigmoid(t * (a - x[:, 1]))


def line_indicator(a, b, c, t=1e5, eps=0., line_type='line'):
    if line_type == 'line':
        return lambda x: torch.sigmoid(-t * (a * x[:, 0] + b * x[:, 1] + c + eps))
    else:
        assert line_type == 'circle'
        center = torch.tensor([a, b]).to(DEVICE)
        return lambda x: torch.sigmoid(-t * (((x - center) ** 2).sum(dim=1) - (c ** 2 + eps)))


def shape_indicator_fn(lines, line_types, eps = 0.0001, boundary_val = 1):
    indicators1 = [line_indicator(*line, line_type=line_type) for line, line_type in zip(lines, line_types)]
    indicators2 = [line_indicator(*line, line_type=line_type, eps=eps) for line, line_type in zip(lines, line_types)]

    return lambda x: boundary_val * prod([indicator(x) for indicator in indicators1]) * (1 - prod([indicator(x) for indicator in indicators2]))

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
                 epsilon_init='const', shape_function=None):
        super().__init__()
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
            # nice_print('indicator min max', ind.min(), ind.max())
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

    # def square_boundary_loss(self, N=128, eps=0.01):
    #     boundary = get_square_boundary(self.a, N + 1).to(DEVICE)
    #     boundary_values = self(boundary)
    #     zero = torch.zeros_like(boundary_values).to(DEVICE)
    #
    #     boundary2 = get_square_boundary((1 + eps) * self.a, N + 1).to(DEVICE)
    #     boundary_values2 = self(boundary2)
    #
    #     term1 = F.l1_loss(torch.minimum(boundary_values, zero), zero, reduction='sum')
    #     term2 = F.l1_loss(torch.maximum(boundary_values2, zero), zero, reduction='sum')
    #     return term1 + term2


    '''
    Compute loss for equation I / A = C. 
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
        nice_print('losses', C_loss.item(), 'A, I', A.item(), I.item(), 'I/A', (I / A).item(), 'C', C.item(), 'logs', np.log((I / A).item()), np.log(C.item()))
        nice_print('entropy', entropy_loss.item(), 'repulsion', repulsion_loss.item(), 'weight decay', weight_decay_loss.item(), 'boundary', boundary_loss.item())
        A_eval, I_eval = self.integrate_eval(N=N)
        C_eval = (I_eval / A_eval).item()
        C_continuous = (I / A).item()
        nice_print('Actual I / A:', np.log(C_eval), 'C', np.log(C.item()))
        combined_loss = C_loss + A_weight * A_loss + entropy_weight * entropy_loss + repulsion_weight * repulsion_loss + weight_decay * weight_decay_loss + boundary_weight * boundary_loss
        return combined_loss, C_eval, C_continuous


def draw_circle(image, r, scale=1):
    h, w = image.shape
    assert h == w
    x, y = np.arange(-h // 2, h // 2, dtype=int), np.arange(-h // 2, h // 2, dtype=int)
    xg, yg = np.meshgrid(x, y)
    rs = xg ** 2 + yg ** 2
    image[np.abs(rs - r ** 2) < 100 * scale] = 1


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
# def set_lr(optim, factor):
#     for g in optim.param_groups:
#         old_lr = float(g['lr'])
#         g['lr'] = factor * old_lr
def midi2hz(m):
    return 440 * 2 ** ((m - 69) / 12)
def hz2midi(f):
    return 12 * np.log2(f / 440) + 69


def run_optimization():
    # L = 0.1
    # L = 0.2
    # L = 0.096258449640988
    # L = 0.13613000497529282
    # a = 0.005
    # C = (a ** 2) / 12
    # a = 0.0025
    # C = (a ** 2) / 4
    E = 200e9
    rho = 7.85e3
    # L = compute_L(256, C, E, rho)
    # print('new L', L)
    L = 0.14
    a = 0.0025

    N = 256
    save_dir = 'exp'

    basis_inits = ['grid']
    lambdas_inits = ['uniform']
    epsilon_inits = ['const']
    A_weights = [0.]
    sigmoid_ts = [1e0]
    entropy_weights = [0.]
    # repulsion_weights = [0.1]
    repulsion_weights = [0., 0.0001]
    repulsion_epsilons = [0.001]
    weight_decays = [0.001]
    boundary_weights = [0., 0.0001]

    # Middle C (C4) is midi number 60, so use notes: C, E, F, G, A, C (next octave).
    # frequencies = [midi2hz(m) for m in [60, 65, 67, 69, 72]]
    frequencies = [midi2hz(m) for m in [60]]
    # frequencies = [compute_f(L, E, rho, (2 * a) ** 4)]
    print('freqs', frequencies)


    combinations = itertools.product(frequencies, basis_inits, lambdas_inits, epsilon_inits, A_weights, sigmoid_ts, entropy_weights, repulsion_weights, repulsion_epsilons, weight_decays, boundary_weights)
    for frequency, basis_init, lambdas_init, epsilon_init, A_weight, sigmoid_t, entropy_weight, repulsion_weight, repulsion_epsilon, weight_decay, boundary_weight in combinations:
        C = compute_C(frequency, L, E, rho)
        # C = compute_C(440, L, E, rho)

        params = {'C': C, 'N': N,
                'A_weight': A_weight,
                'sigmoid_t': sigmoid_t,
                'entropy_weight': entropy_weight,
                'repulsion_weight': repulsion_weight, 'rep_epsilon': repulsion_epsilon,
                'weight_decay': weight_decay, 'boundary_weight': boundary_weight
                }
        print('params:', params)
        model_str = '#'.join('#'.join(str(el) for el in elem) for elem in params.items() if elem[0] != 'C')
        # model_str = '1e-5o.n,e'

        model_dir = save_dir + '/' + model_str + '/sdf'
        image_dir = save_dir + '/' + model_str + '/images'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # shape_function = [[[1, 0, -a], [0, 1, -a], [-1, 0, -a], [0, -1, -a]], ['line', 'line', 'line', 'line']]
        # shape_function = [[[1, 0, -a * 0.8], [0, 1, -a * 0.8], [-1, 0, -a * 0.8], [0, -1, -a * 0.8]], ['line', 'line', 'line', 'line']]

        # shape_function=[[[1, 1, -0.9 * a], [-1, 2, -0.4 * a], [1, -1, -1.2 * a], [-1, -1, -0.8 * a]], ['line', 'line', 'line', 'line']]
        # shape_function=[[[1, 0, 0], [0, -1, 0], [0, 0, a]], ['line', 'line', 'circle']]

        shape_function=[[[0, 0, 1.2 * a]], ['circle']]




        indicator_fn = shape_indicator_fn(*shape_function)
        # indicator_fn = None


        sdf = SDF(1600, a=2 * a, indicator_fn=indicator_fn, basis_init=basis_init,
                # shape_function=[[[1, 1, -a], [-1, 1, -a], [1, -1, -a], [-1, -1, -a]], ['line', 'line', 'line', 'line']]
                # shape_function=[[[0, 0, 0.95 * a]], ['circle']]
                shape_function=shape_function
                # shape_function=[[[1, 0, -0.7 * a], [0, 1, -0.7 * a], [-1, 0, -0.7 * a], [0, -1, -0.7 * a]], ['line', 'line', 'line', 'line']]

                # shape_function=None
                ).to(DEVICE)
        iters = 500
        optimizer = optim.Adam(sdf.parameters(), lr=1e-6)
        total_loss = 0.
        lr_change = {100: 1e-6}
        best_C_dist = float('inf')
        best_C_model = None
        best_C_image = None
        no_improvement = 0
        curr_lr = 1e-5

        for i in range(iters):
            print('iter', i)
            if i in lr_change:
                print('reducing lr to', lr_change[i])
                set_lr(optimizer, lr_change[i])

            S = sdf.get_set()
            optimizer.zero_grad()
            loss, C_eval, C_continuous = sdf.compute_loss(**params)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            S_disc = (S > 0).astype(np.uint8) * 255
            S = (((S - S.min()) / (S.max() - S.min())) * 256).astype(np.uint8)
            if abs(C_eval - C) < best_C_dist:
                best_C_dist = abs(C_eval - C)
                no_improvement = 0
                best_C_image = S_disc
                save_pth = model_dir + '/best.pt'.format(i)
                torch.save(sdf.state_dict(), save_pth)

                img = pil_img.fromarray(S_disc)
                save_pth = image_dir + '/best.pt'.format(i)
                img.save(save_pth.replace('.pt', '') + '_{}_f{}.png'.format(i, hz2midi(frequency)))
                img_continuous = pil_img.fromarray(S)
                img_continuous.save(save_pth.replace('.pt', '') + '_smooth_{}.png'.format(i))

                f_target = compute_f(L, E, rho, C)
                f_model = compute_f(L, E, rho, C_eval)
                m_target = hz2midi(f_target)
                m_model = hz2midi(f_model)
                print('improved, estimated error in hz:', f_target, 'vs', f_model, 'in tones:', m_target, 'vs', m_model)
                if i > 1:
                    plt.clf()
                f, axarr = plt.subplots(1, 2)
                axarr[0].imshow(S, cmap='gray')
                axarr[1].imshow(S_disc, cmap='gray')
                plt.show()

            else:
                print('Not Improved {} epochs'.format(no_improvement))
                improved = False
                no_improvement += 1
                if no_improvement >= 20:
                    # curr_lr *= 0.5
                    # if curr_lr < 1e-6:
                    #     break
                    # set_lr(optimizer, curr_lr)
                    print('reduced learning rate to', curr_lr)
                    no_improvement = 0
    return sdf
if __name__ == '__main__':
    # L = 0.1
    # L = 0.2
    # L = 0.096258449640988
    # L = 0.13613000497529282
    # a = 0.005
    # C = (a ** 2) / 12
    # a = 0.0025
    # C = (a ** 2) / 4
    E = 200e9
    rho = 7.85e3
    # L = compute_L(256, C, E, rho)
    # print('new L', L)
    L = 0.14
    # a = 0.005
    # a = 0.01 / (3 ** 0.5)
    # C = compute_C(256, L, E, rho)
    # print('new C', C)
    # a_sq = ((12 * C) ** 0.5)
    # a_circ = 2 * (C ** 0.5)
    # a = a_sq / 2
    # print('new a', a_sq, a_circ)
    a = 0.0025

    # indicator_fn1=square_indicator(a)
    # indicator_fn2=square_indicator(0.99 * a)
    # indicator_fn = lambda x: indicator_fn1(x) * (1 - indicator_fn2(x))
    # indicator_fn=square_indicator(a)
    # indicator_fn = None
    N = 256
    save_dir = '0101run'


    # basis_inits = ['grid', 'uniform', 'normal']
    # lambdas_inits = ['uniform', 'normal']
    # epsilon_inits = ['uniform', 'normal']
    # A_weights = [0.0001, 0.001, 0.01, 0.1, 1., 10.]
    # sigmoid_ts = [1e0, 1e1, 1e2]
    # entropy_weights = [0.0001, 0.001, 0.01, 0.1, 1., 10.]
    # repulsion_weights = [0.0001, 0.001, 0.01, 0.1, 1., 10.]
    # repulsion_epsilons = [0.0001, 0.001, 0.01, 0.1]
    # weight_decays = [0., 0.1, 0.01, 0.001, 0.0001, 0.00001]
    # boundary_weights = [0., 0.1, 0.01, 0.001, 0.0001, 0.00001]

    basis_inits = ['grid']
    lambdas_inits = ['uniform']
    epsilon_inits = ['const']
    A_weights = [0.]
    sigmoid_ts = [1e0]
    entropy_weights = [0.]
    # repulsion_weights = [0.1]
    repulsion_weights = [0., 0.0001]
    repulsion_epsilons = [0.001]
    weight_decays = [0.001]
    boundary_weights = [0., 0.0001]

    # Middle C (C4) is midi number 60, so use notes: C, E, F, G, A, C (next octave).
    # frequencies = [midi2hz(m) for m in [60, 65, 67, 69, 72]]
    frequencies = [midi2hz(m) for m in [60]]
    # frequencies = [compute_f(L, E, rho, (2 * a) ** 4)]
    print('freqs', frequencies)


    combinations = itertools.product(frequencies, basis_inits, lambdas_inits, epsilon_inits, A_weights, sigmoid_ts, entropy_weights, repulsion_weights, repulsion_epsilons, weight_decays, boundary_weights)
    for frequency, basis_init, lambdas_init, epsilon_init, A_weight, sigmoid_t, entropy_weight, repulsion_weight, repulsion_epsilon, weight_decay, boundary_weight in combinations:
        C = compute_C(frequency, L, E, rho)
        # C = compute_C(440, L, E, rho)

        params = {'C': C, 'N': N,
                'A_weight': A_weight,
                'sigmoid_t': sigmoid_t,
                'entropy_weight': entropy_weight,
                'repulsion_weight': repulsion_weight, 'rep_epsilon': repulsion_epsilon,
                'weight_decay': weight_decay, 'boundary_weight': boundary_weight
                }
        print('params:', params)
        model_str = '#'.join('#'.join(str(el) for el in elem) for elem in params.items() if elem[0] != 'C')
        # model_str = '1e-5o.n,e'

        model_dir = save_dir + '/' + model_str + '/sdf'
        image_dir = save_dir + '/' + model_str + '/images'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # shape_function = [[[1, 0, -a], [0, 1, -a], [-1, 0, -a], [0, -1, -a]], ['line', 'line', 'line', 'line']]
        # shape_function = [[[1, 0, -a * 0.8], [0, 1, -a * 0.8], [-1, 0, -a * 0.8], [0, -1, -a * 0.8]], ['line', 'line', 'line', 'line']]

        # shape_function=[[[1, 1, -0.9 * a], [-1, 2, -0.4 * a], [1, -1, -1.2 * a], [-1, -1, -0.8 * a]], ['line', 'line', 'line', 'line']]
        # shape_function=[[[1, 0, 0], [0, -1, 0], [0, 0, a]], ['line', 'line', 'circle']]

        shape_function=[[[0, 0, 1.2 * a]], ['circle']]




        indicator_fn = shape_indicator_fn(*shape_function)
        # indicator_fn = None


        sdf = SDF(1600, a=2 * a, indicator_fn=indicator_fn, basis_init=basis_init,
                # shape_function=[[[1, 1, -a], [-1, 1, -a], [1, -1, -a], [-1, -1, -a]], ['line', 'line', 'line', 'line']]
                # shape_function=[[[0, 0, 0.95 * a]], ['circle']]
                shape_function=shape_function
                # shape_function=[[[1, 0, -0.7 * a], [0, 1, -0.7 * a], [-1, 0, -0.7 * a], [0, -1, -0.7 * a]], ['line', 'line', 'line', 'line']]

                # shape_function=None
                ).to(DEVICE)
        iters = 500
        optimizer = optim.Adam(sdf.parameters(), lr=1e-6)
        total_loss = 0.
        lr_change = {100: 1e-6}
        best_C_dist = float('inf')
        best_C_model = None
        best_C_image = None
        no_improvement = 0
        curr_lr = 1e-5

        for i in range(iters):
            print('iter', i)
            if i in lr_change:
                print('reducing lr to', lr_change[i])
                set_lr(optimizer, lr_change[i])

            S = sdf.get_set()
            optimizer.zero_grad()
            loss, C_eval, C_continuous = sdf.compute_loss(**params)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            S_disc = (S > 0).astype(np.uint8) * 255
            S = (((S - S.min()) / (S.max() - S.min())) * 256).astype(np.uint8)
            if abs(C_eval - C) < best_C_dist:
                best_C_dist = abs(C_eval - C)
                no_improvement = 0
                best_C_image = S_disc
                save_pth = model_dir + '/best.pt'.format(i)
                torch.save(sdf.state_dict(), save_pth)

                img = pil_img.fromarray(S_disc)
                save_pth = image_dir + '/best.pt'.format(i)
                img.save(save_pth.replace('.pt', '') + '_{}_f{}.png'.format(i, hz2midi(frequency)))
                img_continuous = pil_img.fromarray(S)
                img_continuous.save(save_pth.replace('.pt', '') + '_smooth_{}.png'.format(i))

                f_target = compute_f(L, E, rho, C)
                f_model = compute_f(L, E, rho, C_eval)
                m_target = hz2midi(f_target)
                m_model = hz2midi(f_model)
                print('improved, estimated error in hz:', f_target, 'vs', f_model, 'in tones:', m_target, 'vs', m_model)
                if i > 1:
                    plt.clf()
                f, axarr = plt.subplots(1, 2)
                axarr[0].imshow(S, cmap='gray')
                axarr[1].imshow(S_disc, cmap='gray')
                plt.show()

            else:
                print('else')
                improved = False
                no_improvement += 1
                if no_improvement >= 20:
                    # curr_lr *= 0.5
                    # if curr_lr < 1e-6:
                    #     break
                    # set_lr(optimizer, curr_lr)
                    print('reduced learning rate to', curr_lr)
                    no_improvement = 0