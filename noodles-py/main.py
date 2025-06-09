import noodles_py
import matplotlib.pyplot as plt
import numpy as np

def main():
    heart = "(x^2 + y^2 - 1.)^3 - x^2 * y^3"

    grid = noodles_py.SizedGrid(
        [-2.5, -2.5],
        [5.0, 5.0],
        [500, 500]
    )

    res = noodles_py.contour(heart, grid)

    points = res.points
    edges = res.edges

    points = np.array(points)

    min_x = np.min(points[:,0])
    max_x = np.max(points[:,0])

    min_y = np.min(points[:,1])
    max_y = np.max(points[:,1])

    fig = plt.figure()
    ax = fig.add_subplot()

    for e in edges:
        start = points[e[0]]
        end = points[e[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k')

    ax.set_aspect('equal')
    plt.show()

    return

if __name__ == "__main__":
    main()