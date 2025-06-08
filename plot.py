import matplotlib.pyplot as plt
import numpy as np

import argparse
import os

def fun(X, Y):
    C = (X - 2.5) ** 2 + (Y - 2.5)**2 - (13./8.)**2
    c = (X - 2.5) ** 2 + (Y - 2.5)**2 - (5./8.)**2
    return np.maximum(C, -c)

def fun2(X,Y):
    #(x2+y2−1)3−x2y3=0   
    return (X**2 + Y**2 - 1)**3 - X**2 * Y**3

def read_obj(file_name):
    vertices = []
    edges = []
    with open(file_name, 'r') as in_file:
        for line in in_file.readlines():
            tokens = line.split()
            if tokens[0] == 'v':
                vertices.append([float(tokens[1]), float(tokens[2])])
            elif tokens[0] == 'l':
                edges.append([int(tokens[1]) - 1, int(tokens[2]) - 1])
    return (vertices, edges)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=False)

    args = parser.parse_args()

    obj = None

    if args.file and os.path.isfile(args.file):
        obj = read_obj(args.file)

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    xticks = np.linspace(-5, 5, 5)
    yticks = np.linspace(-5, 5, 5)
    X, Y = np.meshgrid(x, y)
    Z = fun2(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(X, Y, Z, [0.])
    if obj is not None:
        vertices, edges = obj
        for edge in edges:
            ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]], [vertices[edge[0]][1], vertices[edge[1]][1]],'b')
    ax.set_aspect('equal')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid()
    plt.show()
    return

if __name__ == "__main__":
    main()