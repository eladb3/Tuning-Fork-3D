import numpy as np

def midi2hz(m):
    return 440 * 2 ** ((m - 69) / 12)
def hz2midi(f):
    return 12 * np.log2(f / 440) + 69

def get_line(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    # return [y2 - y1, -x2 + x1, -x1 * y2 + x2 * y1]
    return [-y2 + y1, x2 - x1, x1 * y2 - x2 * y1]


def angle2pt(angle):
    return (np.sin(angle), np.cos(angle))


def get_example_shape(shape_type):
    assert shape_type in ['triangle', 'circle', 'square']
    if shape_type == 'triangle':
        a = 0.0025
        angles = [np.pi * (0.5 - 2 / 3), np.pi * 0.5, np.pi * (0.5 + 2 / 3)]
        pts = [angle2pt(angle) for angle in angles]
        shape_scale = 1.6 * a
        pts = [(shape_scale * x, shape_scale * y) for x, y in pts]
        shape_function = [[get_line(pts[0], pts[1]), get_line(pts[1], pts[2]), get_line(pts[2], pts[0])], ['line', 'line', 'line']]
        shape_boundary_val = 0.01

    elif shape_type == 'circle':
        a = 0.0025
        shape_function = [[[0, 0, 1.2 * a]], ['circle']]
        shape_boundary_val = 200

    else: # square
        a = 0.0025
        shape_function = [[[1, 0, -a], [0, 1, -a], [-1, 0, -a], [0, -1, -a]], ['line', 'line', 'line', 'line']]
        shape_boundary_val = 1.5

    return shape_function, shape_boundary_val
