from agent_utils import eval_actions
from agent_utils import select_gpus
from models.PPO_Actor1 import Expert_Actor, GPU_Encoder, MLPCritic
from copy import deepcopy
import torch
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from validation import validate
import os
from Params import configs
from PPO import PPO
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from memory import Memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(epochs):
    from uniform_instance import Simulate_Dataset
    from Simulate_Env import Simulate_Env

    log = []
    # 初始化PPO算法的参数, 并配置与 动态调度 问题相关的参数
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_moe_layer = configs.n_moe_layer,
              n_e=configs.n_e,
              n_g=configs.n_g,
              GCN_num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              GCN_num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    # 这里是随机生成的样本，需修改模拟器！
    simu_tokens = 50 # 假设每对专家之间最多有50个token
    n_e_per_layer = configs.n_e / configs.n_moe_layer
    train_dataset = Simulate_Dataset(n_e_per_layer, configs.n_moe_layer, simu_tokens, configs.num_ins, 200)
    validat_dataset = Simulate_Dataset(n_e_per_layer, configs.n_moe_layer, simu_tokens, 64, 200)
    
    # [样本数, 专家数, 专家数]: 样本k中，专家i到专家j需要路由的token数量
    data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

    record = 1000000
    #开始训练+调度过程
    for epoch in range(epochs):
        memory = Memory() # 存储过程数据
        ppo.policy_old_expert.train()
        ppo.gpu_encoder.train() #最后应该改为encoder+decoder的形式
        ppo.policy_critic.train()
        times, losses, rewards2, critic_rewards = [], [], [], []
        start = time.time()

        costs = []
        expert_losses, gpu_losses, rewards, critic_loss = [], [], [], []
        for batch_idx, batch in enumerate(data_loader):
            env = Simulate_Env(configs.n_moe_layer, configs.n_e, configs.n_g)
            data = batch.numpy() # torch.Size([64, 32, 32])

            # env.reset函数
            expert_links, expert_nodes, expert_adj, gpu_links, gpu_nodes, mask_expert, mask_gpu = env.reset(data)

            expert_log_prob = []
            gpu_log_prob = []
            env_rewards = []
            env_done = [] # 若所有GPU负载已满，则结束动态调度?(显存吃满？还是达到一个较高的吞吐？需要在负载均衡和训练时间上做trade-off)

            gpu_select = []
            pool = None
            ep_rewards = - env.initQuality
            
            env_mask_expert = torch.from_numpy(np.copy(mask_expert)).to(device)
            env_mask_gpu = torch.from_numpy(np.copy(mask_gpu)).to(device)
            while True:
                env_expert_links = deepcopy(expert_links).to(device) # torch.Size([batch siez, n_e, n_e])
                env_expert_nodes = deepcopy(expert_nodes).to(device) # torch.Size([batch siez, n_e, fea_dim = 2])
                env_expert_adj = deepcopy(expert_adj).to(device) # torch.Size([batch siez, n_e, n_e])
                print("env_expert_nodes.shape = ", env_expert_nodes.shape)

                env_gpu_links = deepcopy(gpu_links).to(device) # torch.Size([batch siez, n_g, n_g, fea_dim = 2])
                env_gpu_nodes = deepcopy(gpu_nodes).to(device) # torch.Size([batch siez, n_g, fea_dim = 3])

                # Expert_Actor: 选择需要调度的专家
                expert_probs, expert_select = ppo.policy_old_expert(
                                                        node_features = env_expert_nodes,
                                                        adj_matrix = env_expert_adj,
                                                        mask = env_mask_expert)

                print("Expert Selection[batch 0] = ", expert_select[0], "\nExpert Prob[batch 0] = ", expert_probs[0], "\n")
                expert_indices_tensor = expert_select.clone().detach()
                selected_expert_features = env_expert_nodes[:, expert_indices_tensor[0], :] # torch.Size([64, 2])
                selected_expert_links = env_expert_links[:, expert_indices_tensor[0], :] # torch.Size([64, 32])
                
                # GPU_Actor: 生成 expert_action-gpu_selection 放置决策
                gpu_actions = ppo.gpu_encoder(env_gpu_nodes, env_gpu_links)
                gpu_bool_array, gpu_prob = ppo.policy_old_gpu(
                                                expert_node = selected_expert_features, 
                                                expert_links = selected_expert_links,
                                                gpu_nodes = env_gpu_nodes, 
                                                gpu_links = env_gpu_links, 
                                                pooling_type = configs.graph_pool_type,
                                                mask_gpu_action = env_mask_gpu)
                # print("GPU Selection[batch 0] = ", gpu_bool_array[0], "\nGPU Prob[batch 0] = ", gpu_prob[0], "\n")

                # 记录过程数据
                memory.expert_selection.append(expert_indices_tensor)
                memory.gpu_selection.append(gpu_bool_array)

                memory.expert_node_fea.append(env_expert_nodes)
                memory.expert_link_fea.append(env_expert_links)
                expert_log_prob.append(expert_probs)
                
                memory.gpu_node_fea.append(gpu_nodes)
                memory.gpu_link_fea.append(gpu_links)
                gpu_log_prob.append(gpu_prob)
                

                # 向环境提交选择的动作和机器，接收新的状态、奖励和完成标志等信息, 待修改！！！
                #Simulate_Env里面
                expert_nodes, expert_links,  gpu_nodes, gpu_links, mask_expert, mask_gpu, dur_time, gpu_done, reward = env.step(expert_indices_tensor.numpy(),
                                                                                                gpu_bool_array,
                                                                                                data)
                ep_rewards += reward

                env_rewards.append(deepcopy(reward))
                env_done.append(deepcopy(gpu_done))

                if env.done(): # mask_gpu 没有可用的GPU时，结束
                    break
            
            memory.mask_expert.append(env_mask_expert)
            memory.mask_gpu.append(env_mask_gpu)

            memory.expert_logprobs.append(expert_log_prob)
            memory.gpu_logprobs.append(gpu_log_prob)

            print("memory.env_done = ", env_done, "\n")
            memory.env_rewards.append(torch.tensor(env_rewards).float().permute(1, 0))
            memory.env_done.append(torch.tensor(env_done).float().permute(1, 0))

            # rewards
            ep_rewards -= env.posRewards
            # ppo.update 等待修改
            torch.autograd.set_detect_anomaly(True)
            expert_loss, gpu_loss, value_loss = ppo.update(memory, batch_idx)

            memory.clear_memory()
            mean_time = np.mean(ep_rewards)
            log.append([batch_idx, mean_time])

            # 定期日志记录
            if batch_idx % 100 == 0:
                file_writing_obj = open(
                    './' + 'log_' + str(configs.n_e) + '_' + str(configs.n_g) + '_' + str(configs.low) + '_' + str(
                        configs.high) + '.txt', 'w')
                file_writing_obj.write(str(log))

            rewards.append(np.mean(ep_rewards).item())
            expert_losses.append(expert_loss)
            gpu_losses.append(gpu_loss)
            critic_loss.append(value_loss)

            cost = time.time() - start
            costs.append(cost)
            step = 20

            filepath = 'saved_network'
            # 定期模型保存
            if (batch_idx + 1) % step  == 0 :
                end = time.time()
                times.append(end - start)
                start = end
                expert_mean_loss = np.mean(expert_losses[-step:])
                gpu_mean_loss = np.mean(gpu_losses[-step:])
                mean_time = np.mean(costs[-step:])
                critic_losss = np.mean(critic_loss[-step:])

                filename = 'MultiPPO_{}'.format('Experts'+str(configs.n_e)+'GPU'+str(configs.n_g))
                filepath = os.path.join(filepath, filename)
                epoch_dir = os.path.join(filepath, '%s_%s' % (100, batch_idx))
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                expert_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_expert'))
                gpu_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_gpu'))

                torch.save(ppo.policy_expert.state_dict(), expert_savePath)
                torch.save(ppo.policy_gpu.state_dict(), gpu_savePath)

                print('  Batch %d/%d, mean_time: %2.3f, expert_loss: %2.4f, gpu_loss: %2.4f,critic_loss:%2.4f,took: %2.4fs' %
                      (batch_idx, len(data_loader), mean_time, expert_mean_loss, gpu_mean_loss, critic_losss,
                       times[-1]))

                # 性能评估与验证，用于实时监控模型的学习进度和性能
                t4 = time.time()
                validation_log = validate(valid_loader, configs.batch_size, ppo.policy_expert, ppo.policy_gpu).mean()
                if validation_log<record: # 保存最佳模型
                    epoch_dir = os.path.join(filepath, 'best_value100')
                    if not os.path.exists(epoch_dir):
                        os.makedirs(epoch_dir)
                    expert_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_expert'))
                    gpu_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_gpu'))
                    torch.save(ppo.policy_expert.state_dict(), expert_savePath)
                    torch.save(ppo.policy_gpu.state_dict(), gpu_savePath)
                    record = validation_log

                print('The validation quality is:', validation_log)
                file_writing_obj1 = open(
                    './' + 'vali_' + str(configs.n_e) + '_' + str(configs.n_g) + '_' + str(configs.low) + '_' + str(
                        configs.high) + '.txt', 'w')
                file_writing_obj1.write(str(validation_log))
                t5 = time.time()
        np.savetxt('./N_%s_M%s_u100'%(configs.n_e,configs.n_g),costs,delimiter="\n")


if __name__ == '__main__':
    total1 = time.time()
    main(1)
    total2 = time.time()

    #print(total2 - total1)
