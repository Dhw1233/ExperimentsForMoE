import torch
from Params import configs

def aggr_obs(obs_mb, n_node):
    # obs_mb is [m, n_nodes_each_state, fea_dim], m is number of nodes in batch
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    # print(idx_mb)
    # print(obs_mb.shape[0])
    adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)
    return adj_batch


def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size*n_nodes, 1),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size * n_nodes, 1),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device).view(-1)

    # 生成行索引，每个batch索引重复n_nodes次
    idx_0 = torch.arange(start=0, end=batch_size,
                         device=device,
                         dtype=torch.long)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size*n_nodes, 1)).squeeze()
    # 生成列索引，从0到batch_size * n_nodes
    idx_1 = torch.arange(start=0, end=n_nodes * batch_size,
                         device=device,
                         dtype=torch.long)

    # 组合行列索引
    idx = torch.stack((idx_0, idx_1))

    # 创建稀疏张量
    graph_pool = torch.sparse_coo_tensor(idx, elem,torch.Size([batch_size, n_nodes*batch_size]), device=device)

    print("graph_pool = ", graph_pool.shape)
    return graph_pool


if __name__ == '__main__':
    print('Go home.')