import torch
import torch.nn as nn
import numpy as np

import pdb, sys, os, time
file_path = os.path.dirname(os.path.abspath(__file__))
# file_name = os.path.join(file_path, 'graph_info.npy')
sys.path.append(os.path.split(file_path)[0])
v_num_dict = []
for i in range(12):
    v_num_dict.append(10*2**(2*i)+2)


def load_graph_info(graph_level):
    if graph_level > 4:
        file_name = os.path.join(file_path, f'graph_info_{graph_level}.npy')
    else:
        file_name = os.path.join(file_path, f'graph_info_4m.npy')
    try:
        graph_info = np.load(file_name, allow_pickle=True).item()
    except FileNotFoundError:
        print("will release soon")

    house = graph_info[f'house_{graph_level}']
    neighbor = graph_info[f'neighbor_{graph_level}']
    angle = graph_info[f'angle_{graph_level}']
    return house, neighbor, angle


def interpolate_prepare(angle, kernel_size):
    if kernel_size == 1:  # 1 * 1 conv
        return torch.zeros(1,1)
    v_num = angle.shape[0]
    itp_mat = torch.zeros(7, v_num, kernel_size)  # 7 * v_num * kernel_size
    return itp_mat.permute(1,0,2)  # v_num * 7 * kernel_size
    # how to generate itp_mat will release soon


def gen_indexes(graph_level, client='conv', kernel_size=9):
    assert client in ['conv', 'pool', 'unpool']
    _, neighbor, angle = load_graph_info(graph_level)  # v_num * 7

    if client == 'conv':
        index = torch.tensor(neighbor, dtype=torch.long)
        angle = torch.tensor(angle, dtype=torch.float)
        itp_mat = interpolate_prepare(angle, kernel_size)
        return index, itp_mat

    elif client == 'pool':
        v_num_prime = v_num_dict[graph_level-1]
        index = torch.tensor(neighbor[:v_num_prime], dtype=torch.long)
        return index


if __name__ == "__main__":
    gl, ks = 1, 9
    index, itp_mat = gen_indexes(gl, ks, client='conv')
