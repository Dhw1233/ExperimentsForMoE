import torch
import random
def get_gpu_capacities(n_g):
    # 返回一个包含每个 GPU 可用容量的列表
    capacities = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        available_memory = total_memory - allocated_memory
        capacities.append(available_memory)
    return capacities

def get_gpu_bandwidths(n_g):
    # 返回 GPU 之间的带宽矩阵
    # 根据 gpu_links矩阵
    bandwidth_matrix = [[_ for _ in range(n_g)] for _ in range(n_g)]
    for i in range(n_g):
        for j in range(n_g):
            bandwidth_matrix[i][j] = random.random(0,100)

    return bandwidth_matrix

def calculate_workload_ratios(capacities, bandwidths, local_gpu_id, expert_gpu_ids):
    total_capacity = sum(capacities[gpu_id] for gpu_id in expert_gpu_ids)
    capacity_ratios = [capacities[gpu_id] / total_capacity for gpu_id in expert_gpu_ids]
    
    total_bandwidth = sum(bandwidths[local_gpu_id][gpu_id] for gpu_id in expert_gpu_ids)
    bandwidth_ratios = [bandwidths[local_gpu_id][gpu_id] / total_bandwidth for gpu_id in expert_gpu_ids]
    
    # 综合考虑容量比例和带宽比例
    combined_ratios = [(capacity_ratios[i] + bandwidth_ratios[i]) / 2 for i in range(len(expert_gpu_ids))]
    return combined_ratios

def dispatch_tokens(tokens, local_gpu_id, expert_gpu_ids,n_g):
    # tokens: 需要分发的 token 列表
    # local_gpu_id: 本地 GPU 的 ID
    # expert_gpu_ids: 拥有相同 expert 副本的 GPU ID 列表

    # 步骤 1：检测本地 GPU 和拥有相同 expert 副本的 GPU 的可用容量
    capacities = get_gpu_capacities(n_g)
    bandwidths = get_gpu_bandwidths(n_g)
    expert_capacities = [capacities[gpu_id] for gpu_id in expert_gpu_ids]

    # 步骤 2：根据可用容量计算工作量分配比例
    ratios = calculate_workload_ratios(expert_capacities)

    # 步骤 3：根据本地 GPU 的可用内存分配工作量
    local_capacity = capacities[local_gpu_id]
    local_ratio = ratios[expert_gpu_ids.index(local_gpu_id)]
    local_token_count = int(local_ratio * len(tokens))

    # 分配给本地 GPU 的 token
    local_tokens = tokens[:local_token_count]

    # 步骤 4：将剩余工作量分配给其他 GPU
    remaining_tokens = tokens[local_token_count:]
    distributed_tokens = {local_gpu_id: local_tokens}
    start_idx = 0

    for i, gpu_id in enumerate(expert_gpu_ids):
        if gpu_id == local_gpu_id:
            continue
        token_count = int(ratios[i] * len(tokens))
        distributed_tokens[gpu_id] = remaining_tokens[start_idx:start_idx + token_count]
        start_idx += token_count

    return distributed_tokens # 根据 distributed_tokens[id] 可以获取到对应 GPU 的 token 列表



if __name__ == "__main__":
    # 示例使用
    tokens = [i for i in range(1000)]  # 假设有1000个 tokens
    local_gpu_id = 0
    expert_gpu_ids = [0, 1, 2, 3]  # 假设有 4 个 GPU 拥有相同的 expert 副本

    distributed_tokens = dispatch_tokens(tokens, local_gpu_id, expert_gpu_ids)
    print(distributed_tokens[0])
    print("go home")