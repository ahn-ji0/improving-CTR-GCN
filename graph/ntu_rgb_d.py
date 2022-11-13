import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25

def get_groups():
  #그룹 별로 나누는 곳. 머리, 왼쪽/오른쪽 팔, 왼쪽/오른쪽 다리 return [[],[],[],...]
  groups = []
  #0
  groups.append([21])
  #1
  groups.append([4,3,21])
  #2
  groups.append([24,25,12,11,10,9,21])
  #3
  groups.append([22,23,8,7,6,5,21])
  #4
  groups.append([20,19,18,17,1,2,21])
  #5
  groups.append([16,15,14,13,1,2,21])
  
  return groups

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        ###여기가 실행됨###
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, tools.get_edges(get_groups()))
        ###
        else:
            raise ValueError()
        return A
