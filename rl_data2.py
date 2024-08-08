import numpy as np
import matplotlib.pyplot as plt

# 设置参数
num_experts = 8  # 专家数量
iterations = 2600  # 迭代次数
start_iteration = 2000  # 起始迭代次数

# 初始化负载历史记录
load_history = []

# 生成每个专家的初始负载
initial_loads = np.random.rand(num_experts)
initial_loads /= np.sum(initial_loads)
load_history.append(initial_loads)

# 定义负载变化的函数
def generate_load_change(iteration, num_experts):
    changes = np.zeros(num_experts)
    
    for i in range(num_experts):
        # 使用较小的随机噪声生成平滑的负载变化
        change = 0.01 * np.sin(0.01 * iteration + np.random.rand() * 2 * np.pi) + 0.005 * np.random.randn()
        changes[i] += change
    
    return changes

# # 生成负载变化历史
# for i in range(1, iterations - start_iteration + 1):
#     # 获取上一次的负载
#     previous_loads = load_history[-1]
    
#     # 生成负载变化
#     load_changes = generate_load_change(i, num_experts)
    
#     # 更新负载
#     new_loads = previous_loads + load_changes
    
#     # 保持负载在0到1之间
#     new_loads = np.clip(new_loads, 0, None)
    
#     # 归一化，使得每个时刻的总负载为1
#     new_loads /= np.sum(new_loads)
    
#     # 记录当前迭代的负载
#     load_history.append(new_loads)

# # 转换为numpy数组以便于绘图
# load_history = np.array(load_history)

# # 绘制专家负载的变化
# plt.figure(figsize=(12, 6))
# plt.stackplot(range(start_iteration, iterations + 1), load_history.T, labels=[f'Expert {i}' for i in range(num_experts)])
# plt.xlabel('Iterations')
# plt.ylabel('Accumulated Token Ratio')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
# plt.title('Token Distribution Across Iterations')
# plt.show()

# plt.savefig('RL_Data2.png')