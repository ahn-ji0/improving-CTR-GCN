import numpy as np

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

#####
def edge2mat(link, num_node,step):
    A = np.zeros((num_node, num_node))
    for i, j in link:
      if step == 0:
        A[j, i] = 1
      else:
        A[j, i] = 1 / 2**(step-1)
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

##
def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

##
def get_edges(groups):
  # 그룹별로 나눈 애들의 inward와 outward edge 구하는 곳

  max_length = 0
  # -1씩 해주기(index의 범위는 0부터 이므로)
  for i in range(len(groups)):
    to_idx = []
    #max 알긴 해야함 최대 몇개씩 건너 뛸지 알아햐 하니까. max_length=7
    if len(groups[i]) > max_length:
      max_length = len(groups[i])
    for j in range(len(groups[i])):
      to_idx.append(groups[i][j]-1)
    groups[i] = to_idx
  
  edges = []

  #[[[identity],[inward],[outward]],[[identity],[inward],[outward]],[[identity],[inward],[outward]],...]
  for step in range(0, max_length):
    #건너는게 i
    identity = []
    inward = []
    outward = []
    for group in groups:
      #identity
      if step == 0:
        if len(group) == 1: # group 0 즉 center 일 때
          identity.extend([(j,j) for j in group])
        else:
          identity.extend([(j,j) for j in group[:-1]])

      if step > 0:
        #inward
        inward.extend([(group[j],group[j+step]) for j in range(len(group)) if j+step < len(group)])
        #outward
        outward.extend([(group[j+step],group[j]) for j in range(len(group)) if j+step < len(group)])

    edges.append([identity,inward,outward])
  
  #[[[idenetity],[inward],[outward]],[[identity],[inward],[outward]],[[identity],[inward],[outward]],...]
  return edges

# edge = [[identity],[inward],[outward]]
def get_graph(num_node,edge,step):
  #아래꺼 내용을 여기로 옮기고
  I = edge2mat(edge[0], num_node, step)
  In = edge2mat(edge[1], num_node, step)
  Out = edge2mat(edge[2], num_node, step)
  
  A = np.stack((I,In,Out))
  return A

##
def get_spatial_graph(num_node, edges):
  #여기서는 다 합치는 것만 수행. 아래꺼 삭제
  A = np.zeros((3,num_node,num_node))
  for i in range(len(edges)):
    # edge = [[identity],[inward],[outward]]
    # A[i] shape = np.shape (3,25,25)
    A = A + get_graph(num_node, edges[i],i)
  
  A[1] = normalize_digraph(A[1])
  A[2] =  normalize_digraph(A[2])

  # A shape = (3,25,25)
  return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A
