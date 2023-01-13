import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mtick
import itertools
import copy
import os
from PIL import Image as pil_img

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


def random_minus_one_one(shape):
    return 2 * (torch.rand(shape) - 0.5)


def square_indicator(a, t=1e10):
    return lambda x: torch.sigmoid(t * (x[:, 0] + a)) * torch.sigmoid(t * (a - x[:, 0])) * torch.sigmoid(t * (x[:, 1] + a)) * torch.sigmoid(t * (a - x[:, 1]))


class SDF(nn.Module):
    '''
    Radial basis function with n elements, in the square [-a, a] X [-a, a]
    '''

    def __init__(self, n=100, a=0.005, indicator_fn=None,
                 basis_init='uniform', lambdas_init='uniform',
                 epsilon=3000,
                 epsilon_init='const',
                 barrier_width = 0.0005):
        super(SDF, self).__init__()
        self.a = a
        self.n = n
        if basis_init == 'grid':
            basis = get_grid(self.a, int(self.n ** 0.5))
        elif basis_init == 'uniform':
            basis = self.a * random_minus_one_one((n, 2))
        elif basis_init == 'normal':
            basis = self.a * torch.randn((n, 2))
        self.basis = nn.Parameter(basis)

        if lambdas_init == 'uniform':
            lambdas = self.a * random_minus_one_one((n,))
        elif lambdas_init == 'normal':
            lambdas = self.a * torch.randn((n,))
        self.lambdas = nn.Parameter(lambdas)

        if epsilon_init == 'const':
            epsilons = epsilon * torch.ones((n,))
        elif epsilon_init == 'normal':
            epsilons = 0.2 * epsilon * torch.randn((n,)) + epsilon
        self.epsilons = nn.Parameter(epsilons)

        self.sigmoid = torch.nn.Sigmoid()
        self.indicator_fn = indicator_fn
        self.barrier_width = barrier_width

    '''
    x of size N X 2 (N 2-dimensional points)
    '''
    def forward(self, x_inp):
        x = x_inp.unsqueeze(1).repeat((1, self.basis.shape[0], 1))
        basis = self.basis.unsqueeze(0).repeat((x.shape[0], 1, 1))
        sq_dist = ((x - basis) ** 2).sum(dim=-1)  # compute squared distance from each element of the basis, for each point
        exp = torch.exp(-(self.epsilons ** 2) * sq_dist)  # use exponent for the each radial function with factors epsilon
        res = (self.lambdas * exp).sum(dim=-1) + self.calc_barrier(x[:, 1, :].squeeze())  # weight the radial functions with lambdas
        # res = (self.lambdas * exp).sum(dim=-1)  # weight the radial functions with lambdas
        # if self.indicator_fn is not None:
        #     ind = self.indicator_fn(x_inp)
        #     res += ind
        return res

    def calc_barrier(self, pts):
        t = 50
        eps_b = self.barrier_width
        max_inner_sqr_rad = self.a - eps_b
        barrier = (1 / t) * (-torch.log10(pts[:, 0] + max_inner_sqr_rad)
                             - torch.log10(-pts[:, 0] + max_inner_sqr_rad)
                             - torch.log10(pts[:, 1] + max_inner_sqr_rad)
                             - torch.log10(-pts[:, 1] + max_inner_sqr_rad))
        barrier[torch.isnan(barrier)] = torch.inf
        return barrier


    '''
    Integrate on the square [-a, a] ^ 2 differentiably with a sigmoid
    '''
    def integrate(self, N=256, t=1e0):
        pts = get_grid(self.a, N + 1).cuda()
        S = self.sigmoid(t * self(pts))
        A = S.mean()
        I = (S * (pts ** 2).sum(axis=-1)).mean()
        return A, I, S

    '''
    Integrate on the square [-a, a] ^ 2 (not differentiable) for evaluation
    '''
    def integrate_eval(self, N=512, threshold=1e-8):
        with torch.no_grad():
            pts = get_grid(self.a, N + 1).cuda()
            S = (self(pts) > threshold).float()
            A = S.mean()
            I = (S * (pts ** 2).sum(axis=-1)).mean()
        return A, I

    '''
    Run the SDF on a grid for evaluation
    '''
    def get_set(self):
        N = 2 * int(self.n ** 0.5)
        pts = get_grid(1.5 * self.a, 2 * N).cuda()
        S = self(pts)
        S = S.reshape((2 * N, 2 * N))
        S = S.detach().cpu().numpy()
        return S

    def repulsion_loss(self, epsilon=0.001):
        expanded1 = self.basis.unsqueeze(1).repeat((1, self.n, 1))
        expanded2 = self.basis.unsqueeze(0).repeat((self.n, 1, 1))
        dist = (torch.abs(expanded1 - expanded2)).sum(dim=-1)  # compute distance for each pair of the basis
        epsilon = torch.full(dist.shape, epsilon).cuda()
        zero = torch.zeros_like(dist).cuda()
        return F.l1_loss(torch.maximum(epsilon - dist, zero), zero, reduction='sum')

    def weight_decay(self):
        return F.l1_loss(torch.abs(self.lambdas), torch.zeros_like(self.lambdas), reduction='sum')

    def boundary_loss(self, N=128, eps=0.01):
        boundary = get_square_boundary(self.a, N + 1).cuda()
        boundary_values = self(boundary)
        zero = torch.zeros_like(boundary_values).cuda()

        boundary2 = get_square_boundary((1 + eps) * self.a, N + 1).cuda()
        boundary_values2 = self(boundary2)

        term1 = F.l1_loss(torch.minimum(boundary_values, zero), zero, reduction='sum')
        term2 = F.l1_loss(torch.maximum(boundary_values2, zero), zero, reduction='sum')
        return term1 + term2


    '''
    Compute loss for equation I / A = C. 
    A_weight - regularization term for A area
    N - grid size for integration
    '''

    def compute_loss(self, C, N=256, eps=1e-15,
                     A_weight=0.1, entropy_weight=1e-4, repulsion_weight=1., weight_decay=0.01,
                     rep_epsilon=1e-3, sigmoid_t=1e0, boundary_weight=0.01):
        A, I, S = self.integrate(N=N, t=sigmoid_t)
        C = torch.tensor(C).cuda()
        loss_f = F.l1_loss
        C_loss = loss_f(torch.log(I + eps) - torch.log(A + eps), torch.log(C + eps))
        A_loss = loss_f(torch.log(A + eps), torch.log(torch.zeros_like(A) + eps))
        entropy_loss = loss_f(S * (1 - S), torch.zeros_like(S).cuda())
        repulsion_loss = self.repulsion_loss(epsilon=rep_epsilon)
        weight_decay_loss = self.weight_decay()
        print('losses', C_loss.item(), 'A, I', A.item(), I.item(), 'I/A', (I / A).item(), 'C', C.item(), 'logs', np.log((I / A).item()), np.log(C.item()))
        print('entropy', entropy_loss.item(), 'repulsion', repulsion_loss.item(), 'weight decay', weight_decay_loss.item())
        A_eval, I_eval = self.integrate_eval(N=N)
        C_eval = (I_eval / A_eval).item()
        C_continuous = (I / A).item()
        print('Actual I / A:', np.log(C_eval), 'C', np.log(C.item()))
        repulsion_weight=0
        entropy_weight=0
        combined_loss = C_loss + A_weight * A_loss + entropy_weight * entropy_loss + repulsion_weight * repulsion_loss + weight_decay * weight_decay_loss
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

# L = 0.1
# L = 0.2
# L = 0.096258449640988
# L = 0.13613000497529282
L = 0.15
a = 0.005
E = 200e9
rho = 7.85e3

# indicator_fn1=square_indicator(a)
# indicator_fn2=square_indicator(0.99 * a)
# indicator_fn = lambda x: indicator_fn1(x) * (1 - indicator_fn2(x))
# indicator_fn=square_indicator(a)
indicator_fn = None
N = 256
save_dir = '0101run'
barrier_width = 0.0005

def midi2hz(m):
    return 440 * 2 ** ((m - 69) / 12)
def hz2midi(f):
    return 12 * np.log2(f / 440) + 69
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

if __name__ == '__main__':
    basis_inits = ['uniform']
    lambdas_inits = ['uniform']
    epsilon_inits = ['const']
    A_weights = [0.]
    # sigmoid_ts = [1e0]
    sigmoid_ts = [50]
    entropy_weights = [0.]
    # repulsion_weights = [0.1]
    repulsion_weights = [0., 0.0001]
    repulsion_epsilons = [0.001]
    weight_decays = [0.]
    boundary_weights = [0.]

    # Middle C (C4) is midi number 60, so use notes: C, E, F, G, A, C (next octave).
    # frequencies = [midi2hz(m) for m in [74, 64, 65, 67, 69, 72, 76]]
    frequencies = [midi2hz(m) for m in [72,74,69]]
    freq_tol = 1e-1

    combinations = itertools.product(frequencies, basis_inits, lambdas_inits, epsilon_inits, A_weights, sigmoid_ts, entropy_weights, repulsion_weights, repulsion_epsilons, weight_decays, boundary_weights)
    for frequency, basis_init, lambdas_init, epsilon_init, A_weight, sigmoid_t, entropy_weight, repulsion_weight, repulsion_epsilon, weight_decay, boundary_weight in combinations:
        C = compute_C(frequency, L, E, rho)

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

        sdf = SDF(1600, a=a, indicator_fn=indicator_fn, barrier_width=barrier_width).cuda()
        iters = 5000
        optimizer = optim.Adam(sdf.parameters(), lr=1e-4)
        total_loss = 0.
        lr_change = {40: 1e-5}
        best_C_dist = float('inf')
        best_C_model = None
        best_C_image = None
        no_improvement = 0
        curr_lr = 1e-4

        for i in range(iters):
            print('iter', i)
            # if i in lr_change:
            #     print('reducing lr to', lr_change[i])
            #     set_lr(optimizer, lr_change[i])

            S = sdf.get_set()
            optimizer.zero_grad()
            loss, C_eval, C_continuous = sdf.compute_loss(**params)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            S_disc = (S > 0).astype(np.uint8)
            # S = (((S - S.min()) / (S.max() - S.min())) * 256).astype(np.uint8)
            if abs(C_eval - C) < best_C_dist:
                best_C_dist = abs(C_eval - C)
                no_improvement = 0
                best_C_image = S_disc
                save_pth = model_dir + '/best_midi_{1}.pt'.format(i, int(hz2midi(frequency)))
                torch.save(sdf.state_dict(), save_pth)

                img = pil_img.fromarray(S_disc)
                save_pth = image_dir + '/best_.pt'.format(i)
                img.save(save_pth.replace('.pt', '') + '_{}.png'.format(i))
                img_continuous = pil_img.fromarray(S)
                img_continuous = img_continuous.convert("L")
                img_continuous.save(save_pth.replace('.pt', '') + '_smooth_{}.png'.format(i))

                f_target = compute_f(L, E, rho, C)
                f_model = compute_f(L, E, rho, C_eval)
                m_target = hz2midi(f_target)
                m_model = hz2midi(f_model)
                print('improved, estimated error in hz:', f_target, 'vs', f_model, 'in tones:', m_target, 'vs', m_model)
                if i > 1:
                    plt.clf()
                f, axarr = plt.subplots(1, 2)
                f.tight_layout(w_pad=3)
                a_with_tol=1.5*sdf.a
                pos0=axarr[0].imshow(S, cmap='gray', extent=[-a_with_tol,a_with_tol,-a_with_tol,a_with_tol], vmin=min(S[S!=np.Inf]), vmax=max(S[S!=np.Inf]))
                pos1=axarr[1].imshow(S_disc, cmap='gray', extent=[-a_with_tol,a_with_tol,-a_with_tol,a_with_tol], vmin=0, vmax=1)
                axarr[0].set_title('SDF', y=1.0, pad=20)
                axarr[1].set_title('SDF with TH', y=1.0, pad=20)


                formatter = mtick.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-1, 1))
                axarr[0].xaxis.set_major_formatter(formatter)
                axarr[0].yaxis.set_major_formatter(formatter)
                axarr[1].xaxis.set_major_formatter(formatter)
                axarr[1].yaxis.set_major_formatter(formatter)

                rect = patches.Rectangle((-sdf.a, -sdf.a), 2*sdf.a, 2*sdf.a, linewidth=1, edgecolor='r', facecolor='none')
                axarr[0].add_patch(rect)
                rect = patches.Rectangle((-sdf.a, -sdf.a), 2 * sdf.a, 2 * sdf.a, linewidth=1, edgecolor='r', facecolor='none')
                axarr[1].add_patch(rect)
                f.colorbar(pos0, ax=axarr[0])
                f.colorbar(pos1, ax=axarr[1])
                plt.show()

            else:
                print('else')
                improved = False
                no_improvement += 1
                if no_improvement >= 20:
                    curr_lr *= 0.5
                    if curr_lr < 1e-6:
                        break
                    set_lr(optimizer, curr_lr)
                    print('reduced learning rate to', curr_lr)
                    no_improvement = 0