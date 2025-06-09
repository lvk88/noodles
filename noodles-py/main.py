import noodles_py
import matplotlib.pyplot as plt
import numpy as np

def main():
    heart = "(x^2 + y^2 - 1.)^3 - x^2 * y^3"

    origin = [-1.5, -1.5]
    sizes = [3.0, 3.0]
    gridsize = [32, 32]

    xticks = np.linspace(origin[0], origin[0] + sizes[0], gridsize[0] + 1)
    yticks = np.linspace(origin[1], origin[1] + sizes[1], gridsize[1] + 1)

    grid = noodles_py.SizedGrid(
        origin,
        sizes,
        gridsize
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
        ax.plot([start[0], end[0]], [start[1], end[1]], 'ko-', markersize='3')

    xlabels = ["" for x in xticks]
    ylabels = ["" for x in xticks]
    xlabels[0] = str(xticks[0])
    xlabels[-1] = str(xticks[-1])

    ylabels[0] = str(yticks[0])
    ylabels[-1] = str(yticks[-1])

    ax.set_xticks(xticks, xlabels)
    ax.set_yticks(yticks, ylabels)
    ax.set_aspect('equal')
    ax.grid()
    plt.show()

    return

if __name__ == "__main__":
    main()