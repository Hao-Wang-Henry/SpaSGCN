import numpy as np
import pyvista as pv
import sys, pdb, os, time, datetime

dataset = np.load('data/cifar10visulize.npy')
v_xyz = np.load('spasgcn/graph_info_5.npy', allow_pickle=True).item()['house_5']
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
labels = [3, 8, 8, 0, 6, 6, 1, 6, 3, 1, 0, 9, 5, 7, 9, 8, 5, 7, 8, 6, 7, 0, 4, 9, 5, 2, 4, 0,
        9, 6, 6, 5, 4, 5, 9, 2, 4, 1, 9, 5, 4, 6, 5, 6, 0, 9, 3, 9, 7, 6, 9, 8, 0, 3, 8, 8, 7,
        7, 4, 6, 7, 3, 6, 3, 6, 2, 1, 2, 3, 7, 2, 6, 8, 8, 0, 2, 9, 3, 3, 8, 8, 1, 1, 7, 2, 5,
        2, 7, 8, 9, 0, 3, 8, 6, 4, 6, 6, 0, 0, 7]


def show_on_sphere(v_xyz, number_to_show):
    for number in number_to_show:
        number = number%100

        print(number, classes[labels[number]])
        data = dataset[number]

        sphere = pv.PolyData(v_xyz)
        # pdb.set_trace()
        sphere.point_arrays['scalars'] = data

        p = pv.Plotter(lighting='none')
        p.add_mesh(sphere, rgb=True, render_points_as_spheres=True, point_size=15)
        p.add_axes()
        p.set_background('white')
        p.show()

if __name__ == '__main__':
    number_to_show = [17,18,20,25]
    show_on_sphere(v_xyz, number_to_show)
