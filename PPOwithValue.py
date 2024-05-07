from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_gpus
from models.PPO_Actor1 import Expert_Actor, GPU_Actor
from copy import deepcopy
import torch
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
from epsGreedyForMch import PredictMch
import os
device = torch.device(configs.device)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR



class Memory:
    # 待修改！
    def __init__(self):
        self.expert_node_fea = [] # expert 特征矩阵
        self.expert_link_fea = [] # expert 邻接矩阵
        self.expert_selection = [] # 上一次调度的 expert indice

        self.gpu_node_fea = [] # gpu 特征矩阵
        self.gpu_link_fea = [] # gpu 邻接矩阵
        self.gpu_selection = []
        
        self.reward = [] # 奖励

        self.gpu_token = [] # gpu负载

    def clear_memory(self):
        del self.expert_link_fea[:]
        del self.expert_node_fea[:]
        del self.expert_selection[:]

        del self.gpu_node_fea[:]
        del self.gpu_link_fea[:]
        del self.gpu_selection[:]
        
        del self.reward[:]


def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_moe_layer, # number of moe layers
                 n_e, # the total number of experts 
                 n_g, # number of gpus
                 num_layers, # for GCNN, number of layers in the neural networks (INCLUDING the input layer), 每一层可能使用一个 MLP 来处理节点的特征
                 neighbor_pooling_type, 
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract, # for GCNN, number of layers in mlps (EXCLUDING the input layer), 指定了每个 MLP 的内部层数
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.policy_expert = Expert_Actor(n_moe_layer=configs.n_moe_layer,
                                    n_e=configs.n_e,
                                    num_layers=configs.num_layers,
                                    learn_eps=False,
                                    neighbor_pooling_type=configs.neighbor_pooling_type,
                                    input_dim=configs.input_dim,
                                    hidden_dim=configs.hidden_dim,
                                    output_dim=configs.output_dim,
                                    num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                                    num_mlp_layers_critic=num_mlp_layers_critic,
                                    hidden_dim_critic=hidden_dim_critic,
                                    device=device)
        self.policy_gpu = GPU_Actor(expert_feature_dim = configs.input_dim, 
                                    gpu_feature_dim = 3, 
                                    num_gpus = configs.n_g,
                                    device = device)
        # self.policy_gpu = GPU_Actor(input_dim_expert=2,
        #                             input_dim_gpu=3,
        #                             n_g=configs.n_g,
        #                             n_e=configs.n_e,
        #                             hidden_size=configs.hidden_dim,
        #                             device=device).to(device)

        self.policy_old_expert = deepcopy(self.policy_expert)
        self.policy_old_gpu = deepcopy(self.policy_gpu)

        self.policy_old_expert.load_state_dict(self.policy_expert.state_dict())
        self.policy_old_gpu.load_state_dict(self.policy_gpu.state_dict())

        self.expert_optimizer = torch.optim.Adam(self.policy_expert.parameters(), lr=lr)
        self.gpu_optimizer = torch.optim.Adam(self.policy_gpu.parameters(), lr=lr)

        self.expert_scheduler = torch.optim.lr_scheduler.StepLR(self.expert_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)
        self.gpu_scheduler = torch.optim.lr_scheduler.StepLR(self.gpu_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)


        self.MSE = nn.MSELoss()

    def update(self,  memories, epoch):
        '''self.policy_expert.train()
        self.policy_gpu.train()'''
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
        rewards_all_env = []

        # 计算折扣奖励并进行标准化
        for i in range(configs.batch_size):
            rewards = []

            discounted_reward = 0
            for reward, is_terminal in zip(reversed((memories.reward[0][i]).tolist())):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)

        rewards_all_env = torch.stack(rewards_all_env, 0)
        for _ in range(configs.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            # 这里的节点数不能使用初始化的configs参数，需要动态获取
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [configs.batch_size, configs.n_e * configs.n_g, configs.n_e * configs.n_g]),
                                     n_nodes=configs.n_e * configs.n_g,
                                     device=device)

            expert_log_prob = []
            gpu_log_prob = []
            val = []
            gpu_select = []
            entropies = []
            job_entropy = []
            mch_entropies = []
            expert_scheduler = LambdaLR(self.expert_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
            gpu_scheduler = LambdaLR(self.gpu_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
            job_log_old_prob = memories.expert_logprobs[0]
            mch_log_old_prob = memories.gpu_logprobs[0]
            env_mask_gpu = memories.mask_gpu[0]
            env_expert_token = memories.expert_token[0]

            pool=None
            for i in range(len(memories.expert_node_fea)):
                env_expert_nodes = memories.expert_node_fea[i]
                env_expert_links = memories.expert_link_fea[i]
                env_candidate = memories.expert_selection[i]
                env_mask_expert = memories.mask_expert[i]


                e_index = memories.action[i]

                old_expert = memories.expert_action[i]
                old_mch = memories.gpu_selection[i]

                a_entropy, v, log_e, expert_feature, _, mask_gpu_action, hx = self.policy_expert(x=env_expert_nodes,
                                                                                           graph_pool=g_pool_step,
                                                                                           padded_nei=None,
                                                                                           expert_links=env_expert_links,
                                                                                           expert_selection=env_candidate
                                                                                           , mask_expert=env_mask_expert
                                                                                           , mask_gpu=env_mask_gpu
                                                                                           , expert_token=env_expert_token
                                                                                           , e_index=e_index
                                                                                           , old_expert=old_expert
                                                                                           ,gpu_pool=pool
                                                                                           , old_policy=False
                                                                                           )
                prob_gpu,pool = self.policy_gpu(expert_feature, hx, mask_gpu_action, policy=True)
                val.append(v)
                dist = Categorical(prob_gpu)
                log_mch = dist.log_prob(old_mch)
                mch_entropy = dist.entropy()

                job_entropy.append(a_entropy)
                mch_entropies.append(mch_entropy)
                # entropies.append((mch_entropy+a_entropy))

                expert_log_prob.append(log_e)
                gpu_log_prob.append(log_mch)

            expert_log_prob, job_log_old_prob = torch.stack(expert_log_prob, 0).permute(1, 0), torch.stack(job_log_old_prob,
                                                                                                     0).permute(1, 0)
            gpu_log_prob, mch_log_old_prob = torch.stack(gpu_log_prob, 0).permute(1, 0), torch.stack(mch_log_old_prob,
                                                                                                     0).permute(1, 0)
            val = torch.stack(val, 0).squeeze(-1).permute(1, 0)
            job_entropy = torch.stack(job_entropy, 0).permute(1, 0)
            mch_entropies = torch.stack(mch_entropies, 0).permute(1, 0)

            job_loss_sum = 0
            job_v_loss_sum = 0
            mch_loss_sum = 0
            mch_v_loss_sum = 0
            for j in range(configs.batch_size):
                job_ratios = torch.exp(expert_log_prob[j] - job_log_old_prob[j].detach())
                mch_ratios = torch.exp(gpu_log_prob[j] - mch_log_old_prob[j].detach())
                advantages = rewards_all_env[j] - val[j].detach() # 采取某个行动相对于平均行动的额外预期回报
                advantages = adv_normalize(advantages)
                job_surr1 = job_ratios * advantages
                job_surr2 = torch.clamp(job_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                job_v_loss = self.MSE(val[j], rewards_all_env[j])
                # 策略损失、价值损失(value loss)、熵损失(鼓励探索)
                job_loss = -1*torch.min(job_surr1, job_surr2) + 0.5*job_v_loss - 0.01 * job_entropy[j]
                job_loss_sum += job_loss


                mch_surr1 = mch_ratios * advantages
                mch_surr2 = torch.clamp(mch_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                #mch_v_loss = self.MSE(val[j], rewards_all_env[j])
                mch_loss = -1*torch.min(mch_surr1, mch_surr2) - 0.01 * mch_entropies[j]
                mch_loss_sum += mch_loss

            # take gradient step
            # loss_sum = torch.stack(loss_sum,0)
            # v_loss_sum = torch.stack(v_loss_sum,0)
            self.expert_optimizer.zero_grad()
            job_loss_sum.mean().backward(retain_graph=True)

            # scheduler.step()
            # Copy new weights into old policy:
            self.policy_old_expert.load_state_dict(self.policy_expert.state_dict())

            self.gpu_optimizer.zero_grad()
            mch_loss_sum.mean().backward(retain_graph=True)
            self.expert_optimizer.step()
            self.gpu_optimizer.step()
            # scheduler.step()
            # Copy new weights into old policy:
            self.policy_old_gpu.load_state_dict(self.policy_gpu.state_dict())
            
            if configs.decayflag:
                self.expert_scheduler.step()
            if configs.decayflag:
                self.gpu_scheduler.step()

            return job_loss_sum.mean().item(), mch_loss_sum.mean().item()


def main(epochs):
    from uniform_instance import Simulate_Dataset
    from Simulate_Env import Simulate_Env

    log = []
    env_expert_pool = g_pool_cal(graph_pool_type = configs.graph_pool_type,
                             batch_size = configs.batch_size,
                             n_nodes = configs.n_e,
                             device = device)
    env_gpu_pool = g_pool_cal(graph_pool_type = configs.graph_pool_type,
                             batch_size = configs.batch_size,
                             n_nodes = configs.n_g,
                             device = device)

    # 初始化PPO算法的参数, 并配置与 动态调度 问题相关的参数
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_moe_layer = configs.n_moe_layer,
              n_e=configs.n_e,
              n_g=configs.n_g,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    # 这里是随机生成的样本，需修改模拟器！
    simu_tokens = 50 # 假设每对专家之间最多有50个token
    n_e_per_layer = configs.n_e / configs.n_moe_layer
    train_dataset = Simulate_Dataset(n_e_per_layer, configs.n_moe_layer, simu_tokens, configs.num_ins, 200)
    validat_dataset = Simulate_Dataset(n_e_per_layer, configs.n_moe_layer, simu_tokens, 64, 200)
    print("Simulate Val-Dataset Success!\n", validat_dataset.getdata()[0,0,:], "\n")

    data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)
    '''ppo.policy_old_expert.to(device)
    ppo.policy_old_gpu.to(device)'''

    record = 1000000
    for epoch in range(epochs):
        memory = Memory() # 存储过程数据
        ppo.policy_old_expert.train()
        ppo.policy_old_gpu.train()

        times, losses, rewards2, critic_rewards = [], [], [], []
        start = time.time()

        costs = []
        losses, rewards, critic_loss = [], [], []
        for batch_idx, batch in enumerate(data_loader):
            env = Simulate_Env(configs.n_moe_layer, configs.n_e, configs.n_g)
            data = batch.numpy()

            # env.reset函数
            expert_links, expert_nodes, gpu_links, gpu_nodes, mask_expert, mask_gpu = env.reset(data)

            expert_log_prob = []
            gpu_log_prob = []
            reward = []

            j = 0
            gpu_select = []
            pool = None
            ep_rewards = - env.initQuality
            
            while True:
                env_expert_links = deepcopy(expert_links).float().to(device)
                env_expert_nodes = torch.from_numpy(np.copy(expert_nodes)).float().to(device)
                print("deepcopy(env_expert_links) \n", env_expert_links.shape)
                print("deepcopy(env_expert_nodes) \n", env_expert_nodes.shape)

                env_mask_expert = torch.from_numpy(np.copy(mask_expert)).to(device)
                env_mask_gpu = torch.from_numpy(np.copy(mask_gpu)).to(device)

                # Expert_Actor: 选择需要调度的专家
                expert_indices = ppo.policy_old_expert(
                                                        expert_nodes = env_expert_nodes,
                                                        expert_links = env_expert_links,
                                                        graph_pool = env_expert_pool,
                                                        padded_nei = None,
                                                        mask_expert = env_mask_expert)
                
                expert_indices_tensor = expert_indices.clone().detach()
                selected_expert_features = env_expert_nodes[:, expert_indices_tensor[0], :] # torch.Size([64, 2])
                selected_expert_links = env_expert_links[:, expert_indices_tensor[0], :] # torch.Size([64, 32])
                
                # GPU_Actor: 生成 expert_action-gpu_selection 放置决策
                gpu_bool_array = ppo.policy_old_gpu(
                                                expert_node = selected_expert_features, 
                                                expert_links = selected_expert_links,
                                                gpu_nodes = gpu_nodes, 
                                                gpu_links = gpu_links, 
                                                mask_gpu_action = env_mask_gpu)
                print("Expert Selection[batch 0] = ", expert_indices[0], "\nGPU Bool Array[batch 0] = ", gpu_bool_array[0])

                # 记录过程数据
                memory.expert_selection.append(expert_indices)
                memory.gpu_selection.append(gpu_bool_array)

                memory.expert_node_fea.append(env_expert_nodes)
                memory.expert_link_fea.append(env_expert_links)
                
                memory.gpu_node_fea.append(gpu_nodes)
                memory.gpu_link_fea.append(gpu_links)
                
                # 向环境提交选择的动作和机器，接收新的状态、奖励和完成标志等信息
                expert_links, expert_nodes, gpu_nodes, gpu_links, reward, dur_time = env.step(expert_indices.cpu().numpy(),
                                                                                               gpu_bool_array)
                ep_rewards += reward
                reward.append(deepcopy(reward))

                j += 1
                if env.done(): # mask_gpu 没有可用的GPU时，结束
                    break
            memory.expert_token.append(env_expert_token)
            memory.mask_gpu.append(env_mask_gpu)

            memory.expert_logprobs.append(expert_log_prob)
            memory.gpu_logprobs.append(gpu_log_prob)
            memory.reward.append(torch.tensor(reward).float().permute(1, 0))
            # -------------------------------------------------------------------------------------
            ep_rewards -= env.posRewards
            # -------------------------------------------------------------------------------------
            loss, v_loss = ppo.update(memory,batch_idx)
            memory.clear_memory()
            mean_reward = np.mean(ep_rewards)
            log.append([batch_idx, mean_reward])

            # 定期日志记录
            if batch_idx % 100 == 0:
                file_writing_obj = open(
                    './' + 'log_' + str(configs.n_e) + '_' + str(configs.n_g) + '_' + str(configs.low) + '_' + str(
                        configs.high) + '.txt', 'w')
                file_writing_obj.write(str(log))

            rewards.append(np.mean(ep_rewards).item())
            losses.append(loss)
            critic_loss.append(v_loss)

            cost = env.mchsEndTimes.max(-1).max(-1)
            costs.append(cost.mean())
            step = 20

            filepath = 'saved_network'
            # 定期模型保存
            if (batch_idx + 1) % step  == 0 :
                end = time.time()
                times.append(end - start)
                start = end
                mean_loss = np.mean(losses[-step:])
                mean_reward = np.mean(costs[-step:])
                critic_losss = np.mean(critic_loss[-step:])

                filename = 'FJSP_{}'.format('J'+str(configs.n_e)+'M'+str(configs.n_g))
                filepath = os.path.join(filepath, filename)
                epoch_dir = os.path.join(filepath, '%s_%s' % (100, batch_idx))
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                job_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_expert'))
                machine_savePate = os.path.join(epoch_dir, '{}.pth'.format('policy_gpu'))

                torch.save(ppo.policy_expert.state_dict(), job_savePath)
                torch.save(ppo.policy_gpu.state_dict(), machine_savePate)

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f,critic_loss:%2.4f,took: %2.4fs' %
                      (batch_idx, len(data_loader), mean_reward, mean_loss, critic_losss,
                       times[-1]))

                # 性能评估与验证，用于实时监控模型的学习进度和性能
                t4 = time.time()
                validation_log = validate(valid_loader, configs.batch_size, ppo.policy_expert, ppo.policy_gpu).mean()
                if validation_log<record: # 保存最佳模型
                    epoch_dir = os.path.join(filepath, 'best_value100')
                    if not os.path.exists(epoch_dir):
                        os.makedirs(epoch_dir)
                    job_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_expert'))
                    machine_savePate = os.path.join(epoch_dir, '{}.pth'.format('policy_gpu'))
                    torch.save(ppo.policy_expert.state_dict(), job_savePath)
                    torch.save(ppo.policy_gpu.state_dict(), machine_savePate)
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
