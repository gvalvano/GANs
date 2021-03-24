import cmath
from math import atan2, pi
import random
import numpy as np
from skimage.morphology import area_closing
import cv2


def generate_random_blobs(n_blobs, n_classes, image_size, offset, min_blob_radii, max_blob_radii=None):

    def _convexHull(points):
        # Graham's scan.
        x_leftmost, y_leftmost = min(points)
        by_theta = [(atan2(x-x_leftmost, y-y_leftmost), x, y) for x, y in points]
        by_theta.sort()
        as_complex = [complex(x, y) for _, x, y in by_theta]
        cvx_hull = as_complex[:2]
        for pt in as_complex[2:]:
            # Perp product.
            while ((pt - cvx_hull[-1]).conjugate() * (cvx_hull[-1] - cvx_hull[-2])).imag < 0:
                cvx_hull.pop()
            cvx_hull.append(pt)
        return [(pt.real, pt.imag) for pt in cvx_hull]

    def _dft(xs):
        return [sum(x * cmath.exp(2j*pi*i*k/len(xs))
                    for i, x in enumerate(xs))
                for k in range(len(xs))]

    def _interpolateSmoothly(xs, N):
        """For each point, add N points."""
        fs = _dft(xs)
        half = (len(xs) + 1) // 2
        fs2 = fs[:half] + [0]*(len(fs)*N) + fs[half:]
        return [x.real / len(xs) for x in _dft(fs2)[::-1]]

    def _filter_allowed(v, v_max):
        return int(max(0, min(v_max - 1, v)))

    width, height = image_size
    delta_x, delta_y = offset
    mask = np.zeros((width, height, n_classes))
    for b in range(n_blobs):
        for c in range(n_classes):

            if max_blob_radii is None:
                blob_radii = min_blob_radii
            else:
                a = min_blob_radii
                b = max_blob_radii
                blob_radii = [(b[i] - a[i]) * np.random.random_sample() + a[i] for i in range(len(min_blob_radii))]
                blob_radii = [int(el) for el in blob_radii]

            x0 = np.random.random_integers(delta_x, width - delta_x)
            y0 = np.random.random_integers(delta_y, height - delta_y)

            pts = [(random.random() + 0.8) * cmath.exp(2j * pi * i / 7) for i in range(7)]
            pts = _convexHull([(pt.real, pt.imag) for pt in pts])
            xs, ys = [_interpolateSmoothly(zs, 10) for zs in zip(*pts)]
            xs = [_filter_allowed(el * blob_radii[0] + x0, width) for el in xs]
            ys = [_filter_allowed(el * blob_radii[1] + y0, height) for el in ys]

            mask[xs, ys, c] = 1
            # mask[..., c] = area_closing(mask[..., c])
            kernel = np.ones((2 * blob_radii[0], 2 * blob_radii[1]))
            mask[..., c] = cv2.morphologyEx(mask[..., c], cv2.MORPH_CLOSE, kernel=kernel)

    return mask


# def generate_random_blobs_v2(n_blobs, n_classes, image_size, offset):
#
#     def get_possible_directions(point):
#         """Point is in form (x, y, z)"""
#         directions = [
#             [point[0] + 1, point[1]],  # point[2]],  # right
#             [point[0] - 1, point[1]],  # point[2]],  # left
#             [point[0], point[1] + 1],  # point[2]],  # forward
#             [point[0], point[1] - 1],  # point[2]],  # backward
#             # [point[0], point[1], point[2] + 1],  # up
#             # [point[0], point[1], point[2] - 1]  # down
#         ]
#         return directions
#
#     def random_walk(n_steps, starting_position=None):
#         if starting_position is None:
#             starting_position = [0, 0]
#         current_position = starting_position
#         visited_points = []
#         for _ in range(n_steps):
#             visited_points.append(current_position)
#             all_directions = get_possible_directions(current_position)
#             not_visited_directions = [direction for direction in all_directions if direction not in visited_points]
#             current_position = random.choice(not_visited_directions)
#
#         # xp, yp = zip(*visited_points)
#         # return xp, yp  # returns tuples. If you want lists, just do list(xp), ...
#         return visited_points
#
#     width, height = image_size
#     delta_x, delta_y = offset
#     mask = np.zeros((width, height, n_classes))
#     for b in range(n_blobs):
#         for c in range(n_classes):
#             x0 = np.random.random_integers(delta_x, width - delta_x)
#             y0 = np.random.random_integers(delta_y, height - delta_y)
#             center = (x0, y0)
#             walk = random_walk(20, starting_position=center)
#             mask[..., c][walk] = 1
#     return mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    size = (224, 224)
    radii = size[0]//100, size[1]//100
    ofs = [size[0]//10, size[1]//10]
    add_mask = generate_random_blobs(n_blobs=5, n_classes=3, image_size=size, offset=ofs, min_blob_radii=radii)
    ofs = [size[0]//4, size[1]//4]
    remove_mask = generate_random_blobs(n_blobs=3, n_classes=3, image_size=size, offset=ofs, min_blob_radii=radii)
    
    plt.figure()
    plt.imshow(add_mask)
    plt.show()
    
    plt.figure()
    plt.imshow(remove_mask)
    plt.show()
