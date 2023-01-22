from tuning_fork import midi2hz, TuningFork
from tuning_fork.utils import *
import numpy as np
# a = 0.0025


if __name__ == '__main__':
    # Pitch numbers are in MIDI scale - 60 is C4 (central C). Units are in semi-tones.
    example_shape, pitch = 'triangle', 63
    # example_shape, pitch = 'triangle', 62
    # example_shape, pitch = 'triangle', 61

    # example_shape, pitch = 'square', 66
    # example_shape, pitch = 'square', 64
    # example_shape, pitch = 'square', 60

    # example_shape, pitch = 'circle', 64
    # example_shape, pitch = 'circle', 63
    # example_shape, pitch = 'circle', 62

    shape_function, shape_boundary_val = get_example_shape(example_shape)
    print('example shape:', example_shape, 'pitch:', pitch)

    tfork = TuningFork(
        shape_function=shape_function,
        shape_boundary_val=shape_boundary_val,
        base_lr=1e-5,
        shape_eps = 0.0001,
        frequency = midi2hz(pitch),
        basis_init = 'grid',
        lambdas_init = 'uniform',
        epsilon_init = 'const',
        A_weight = 0,
        sigmoid_temp = 1e0,
        entropy_weight = 0.,
        weight_decay = 0.001,
        error=1e-3, max_iters=100, verbose=False
    )
    tfork.optimize()
    tfork.N = 32 # Lower resolution for 3D model
    tfork.to_stl(outpath = f"tuning-fork.stl", simplify_mesh=False, close_sides=True)