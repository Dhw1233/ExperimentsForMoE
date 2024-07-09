from agent_utils import eval_actions
from agent_utils import select_gpus
from models.PPO_Actor1 import EXPERT_ACTOR, GPU_ACTOR, MLPCritic
from copy import deepcopy
import torch
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
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
        self.mask_expert = []
        self.expert_logprobs = [] # expert 的 log 概率  

        self.gpu_node_fea = [] # gpu 特征矩阵
        self.gpu_link_fea = [] # gpu 邻接矩阵
        self.gpu_selection = []
        self.mask_gpu = []
        self.gpu_logprobs = [] # gpu 的 log 概率
        
        self.act_selection = [] #gpu add/delete
        self.act_logprobs = [] #act的log概率

        self.env_rewards = [] # 奖励
        self.env_done = []


    def clear_memory(self):
        del self.expert_link_fea[:]
        del self.expert_node_fea[:]
        del self.expert_selection[:]
        del self.mask_expert[:]
        del self.expert_logprobs[:]

        del self.gpu_node_fea[:]
        del self.gpu_link_fea[:]
        del self.gpu_selection[:]
        del self.mask_gpu[:]
        del self.gpu_logprobs[:]
        
        del self.env_rewards[:]
        del self.env_done[:]
        del self.act_logprobs[:]
        del self.act_selection[:]

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
                *args,
                **kwargs
                ):
        self.lr = lr
        
        self.gamma = gamma
        
        self.eps_clip = eps_clip
        
        self.k_epochs = k_epochs

        self.expert_policy = EXPERT_ACTOR(
                                    expert_feature_dim = configs.expert_feature_dim,
                                    gpu_feature_dim = configs.gpu_feature_dim,
                                    hidden_dim = configs.hidden_dim,
                                    expert_output_dim=configs.expert_output_dim,
                                    gpu_output_dim=configs.gpu_output_dim,
                                    num_layers = configs.num_layers, 
                                    num_encoder_layers = configs.num_mlp_layers_feature_extract,
                                    num_decoder_layers = configs.num_mlp_layers_actor,
                                    num_critic_layers  = configs.num_mlp_layers_critic,
                                    num_experts = configs.n_e,
                                    num_gpus = configs.n_g,
                                    old_policy=False).to(device)
        
        
        self.gpu_policy = GPU_ACTOR(
                                    expert_feature_dim = configs.expert_feature_dim,
                                    gpu_feature_dim = configs.gpu_feature_dim,
                                    hidden_dim = configs.hidden_dim,
                                    expert_output_dim=configs.expert_output_dim,
                                    gpu_output_dim=configs.gpu_output_dim,
                                    num_layers = configs.num_layers, 
                                    num_encoder_layers = configs.num_mlp_layers_feature_extract,
                                    num_decoder_layers = configs.num_mlp_layers_actor,
                                    num_critic_layers  = configs.num_mlp_layers_critic,
                                    num_experts = configs.n_e,
                                    num_gpus = configs.n_g,
                                    old_policy=True).to(device)
        
        self.old_expert_policy = EXPERT_ACTOR(
                                    expert_feature_dim = configs.expert_feature_dim,
                                    gpu_feature_dim = configs.gpu_feature_dim,
                                    hidden_dim = configs.hidden_dim,
                                    expert_output_dim=configs.expert_output_dim,
                                    gpu_output_dim=configs.gpu_output_dim,
                                    num_layers = configs.num_layers, 
                                    num_encoder_layers = configs.num_mlp_layers_feature_extract,
                                    num_decoder_layers = configs.num_mlp_layers_actor,
                                    num_critic_layers  = configs.num_mlp_layers_critic,
                                    num_experts = configs.n_e,
                                    num_gpus = configs.n_g,
                                    old_policy=False).to(device)
        
        self.old_gpu_policy = GPU_ACTOR(
                                    expert_feature_dim = configs.expert_feature_dim,
                                    gpu_feature_dim = configs.gpu_feature_dim,
                                    hidden_dim = configs.hidden_dim,
                                    expert_output_dim=configs.expert_output_dim,
                                    gpu_output_dim=configs.gpu_output_dim,
                                    num_layers = configs.num_layers, 
                                    num_encoder_layers = configs.num_mlp_layers_feature_extract,
                                    num_decoder_layers = configs.num_mlp_layers_actor,
                                    num_critic_layers  = configs.num_mlp_layers_critic,
                                    num_experts = configs.n_e,
                                    num_gpus = configs.n_g,
                                    old_policy=True).to(device)

        self.old_expert_policy.load_state_dict(self.expert_policy.state_dict())

        self.old_gpu_policy.load_state_dict(self.gpu_policy.state_dict())

        self.expert_optimizer = torch.optim.Adam(self.expert_policy.parameters(), lr=lr)

        self.gpu_optimizer = torch.optim.Adam(self.gpu_policy.parameters(), lr=lr)

        self.expert_scheduler = torch.optim.lr_scheduler.StepLR(self.expert_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)
    
        self.gpu_scheduler = torch.optim.lr_scheduler.StepLR(self.gpu_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)

        self.MSE = nn.MSELoss()

    def update(self, memories, epoch):
        self.expert_policy.train()
        self.gpu_policy.train()

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
        rewards_all_env = []
        rewards_all_env = torch.tensor(rewards_all_env)  
        cnt1,cnt2=0,0
        # 计算折扣奖励并进行标准化
        for rewards_list, dones_list in zip(memories.env_rewards, memories.env_done):
            rewards = torch.tensor([],dtype=torch.float32)
            discounted_reward = 0
            cnt1+=1
            rewards_list = rewards_list.squeeze() # 转换为一维张量
            dones_list = dones_list.squeeze()
            cnt2=0
            for reward, is_terminal in zip(reversed(rewards_list), reversed(dones_list)):
                if all(is_terminal == 1):
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                cnt2+=1
                if cnt2 == 1:
                    rewards = discounted_reward.unsqueeze(0)
                else:
                    rewards = torch.cat((discounted_reward.unsqueeze(0),rewards),dim=0)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            if cnt1 == 1 :
                rewards_all_env = rewards
            else:
                rewards_all_env=torch.cat((rewards_all_env,rewards.unsqueeze(0)),dim=0)

        rewards_all_env = rewards_all_env.to(torch.float32).squeeze().detach().to(device)
        
        for _ in range(configs.k_epochs):

            expert_log_prob = []
            gpu_log_prob = []
            act_log_prob = []
            val = []


            expert_entropy = []
            gpu_entropy = []
            act_entropy = []

            expert_log_old_prob = memories.expert_logprobs[0]
            gpu_log_old_prob = memories.gpu_logprobs[0]
            act_log_old_prob = memories.act_logprobs[0]

            for i in range(len(memories.expert_node_fea)):
                env_expert_nodes = memories.expert_node_fea[i].float()
                env_expert_links = memories.expert_link_fea[i].float()
                env_gpu_nodes = memories.gpu_node_fea[i].float()
                env_gpu_links = memories.gpu_link_fea[i].float()
                env_mask_expert = memories.mask_expert[i]
                env_mask_gpu = memories.mask_gpu[i]

                expert_prob,expert_index,v=self.expert_policy(ep_nodes=env_expert_nodes,
                                                                            ep_links=env_expert_links,
                                                                            gp_nodes=env_gpu_nodes, 
                                                                            gp_links=env_gpu_links,
                                                                            mask_ep=env_mask_expert,
                                                                            mask_gp=env_mask_gpu,
                                                                            old_policy = False)

                gpu_prob,gpu_index,act_prob,act_index = self.gpu_policy(ep_nodes=env_expert_nodes,
                                                                            ep_links=env_expert_links,
                                                                            gp_nodes=env_gpu_nodes, 
                                                                            gp_links=env_gpu_links,
                                                                            mask_ep=env_mask_expert,
                                                                            mask_gp=env_mask_gpu,
                                                                            ep_index=expert_index,
                                                                            old_policy = False)

                # Combine (action_e, action_g)
                val.append(v.squeeze().to(torch.float32))

                # Calculate the log probabilities
                expert_log_prob.append(torch.log(expert_prob + 1e-10))
                gpu_log_prob.append(torch.log(gpu_prob + 1e-10))
                act_log_prob.append(torch.log(act_prob+1e-10))
                # Calculate the entropies
                expert_dist = Categorical(expert_prob)
                expert_entropy.append(expert_dist.entropy())

                gpu_dist = Categorical(gpu_prob)
                gpu_entropy.append(gpu_dist.entropy())

                act_dist = Categorical(act_prob)
                act_entropy.append(act_dist.entropy())
            # Convert lists to tensors
            expert_log_prob, expert_log_old_prob = torch.cat(expert_log_prob, dim=0), torch.cat(expert_log_old_prob, dim=0) # torch.Size([64, 32])
            gpu_log_prob, gpu_log_old_prob = torch.cat(gpu_log_prob, dim=0), torch.cat(gpu_log_old_prob, dim=0) # torch.Size([64, 4])
            act_log_prob,act_log_old_prob = torch.cat(act_log_prob,dim=0), torch.cat(act_log_old_prob,dim=0)
            val = torch.stack(val,dim=0).permute(1,0).float().to(device) # torch.Size([64])
            val = (val-val.mean())/(val.std() + 1e-8)
            expert_entropy = torch.cat(expert_entropy).squeeze() # torch.Size([64])
            gpu_entropy = torch.cat(gpu_entropy).squeeze() # torch.Size([64])
            act_entropy = torch.cat(act_entropy).squeeze()
            # Compute advantages
            advantages = rewards_all_env - val.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss
            expert_loss_sum = torch.zeros(1, device=device)
            gpu_loss_sum = torch.zeros(1, device=device)
            act_loss_sum = torch.zeros(1,device=device)
            value_loss_sum = torch.zeros(1, device=device)
            for j in range(configs.num_ins):
                for i in range(len(memories.expert_node_fea)):
                    ep_index=memories.expert_selection[i][j]
                    gp_index=memories.gpu_selection[i][j]
                    ac_index=memories.act_selection[i][j]
                    expert_ratios = torch.exp(expert_log_prob[j] - expert_log_old_prob[j].detach()) # torch.Size([32])
                    gpu_ratios = torch.exp(gpu_log_prob[j] - gpu_log_old_prob[j].detach()) #  torch.Size([4])
                    act_ratios = torch.exp(act_log_prob[j] - act_log_old_prob[j].detach())

                    expert_ratios = expert_ratios[ep_index]
                    gpu_ratios = gpu_ratios[gp_index]
                    act_ratios = act_ratios[ac_index]

                    expert_surr1 = expert_ratios * advantages[j] # torch.Size([32])
                    expert_surr2 = torch.clamp(expert_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[j] # torch.Size([32])
                    expert_loss = -1 * torch.min(expert_surr1, expert_surr2) - entloss_coef * expert_entropy[j]
                    expert_loss_sum += expert_loss.sum() # torch.Size([1])

                    gpu_surr1 = gpu_ratios * advantages[j]
                    gpu_surr2 = torch.clamp(gpu_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[j]
                    gpu_loss = -1 * torch.min(gpu_surr1, gpu_surr2) - entloss_coef * gpu_entropy[j]
                    gpu_loss_sum += gpu_loss.sum() # torch.Size([1])

                    act_surr1 = act_ratios * advantages[j]
                    act_surr2 = torch.clamp(act_ratios,1-self.eps_clip,1+self.eps_clip) * advantages[j]
                    act_loss = -1 * torch.min(act_surr1,act_surr2) - entloss_coef * act_entropy[j]
                    act_loss_sum += act_loss.sum()

                value_loss = self.MSE(val[j], rewards_all_env[j])
                value_loss_sum += value_loss # torch.Size([1])

            # Calculate the total loss
            total_expert_loss = ploss_coef * expert_loss_sum / configs.num_ins
            total_gpu_loss = ploss_coef * gpu_loss_sum / configs.num_ins
            total_act_loss = ploss_coef * act_loss_sum / configs.num_ins
            total_value_loss = vloss_coef * value_loss_sum / configs.num_ins

            # take gradient step, scheduler.step()
            self.expert_optimizer.zero_grad()
            total_expert_loss.backward(retain_graph=True)
            total_value_loss.backward(retain_graph=True)
            self.expert_optimizer.step()

            self.gpu_optimizer.zero_grad()
            total_gpu_loss.backward(retain_graph=True)
            total_act_loss.backward(retain_graph=True)
            self.gpu_optimizer.step()

            
            if configs.decayflag:
                self.expert_scheduler.step()
                self.gpu_scheduler.step()

        self.old_expert_policy.load_state_dict(self.expert_policy.state_dict())
        self.old_gpu_policy.load_state_dict(self.gpu_policy.state_dict())
        return expert_loss_sum.mean().item(), gpu_loss_sum.mean().item(), act_loss_sum.mean().item(),value_loss_sum.mean().item()
    

def main(epochs):
    from uniform_instance import Simulate_Dataset
    from Simulate_Env import Simulate_Env

    log = []
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
    n_e_per_layer = configs.n_e // configs.n_moe_layer
    train_dataset = Simulate_Dataset(n_e_per_layer,configs.n_e, configs.n_g,configs.n_moe_layer, simu_tokens,configs.batch_num,configs.batch_size, configs.num_ins, 200)
    validat_dataset = Simulate_Dataset(n_e_per_layer,configs.n_e, configs.n_g,configs.n_moe_layer, simu_tokens,configs.batch_num,configs.batch_size, configs.num_ins, 200)
    # print("Simulate Val-Dataset Success!\n", validat_dataset.getdata()[0,0,:], "\n")
    SampleCnt = configs.num_ins 
    StateCnt = 10
    data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

    record = 1000000
    for epoch in range(epochs):
        memory = Memory() # 存储过程数据
        ppo.old_expert_policy.train()
        ppo.old_gpu_policy.train()

        times, losses, rewards2, critic_rewards = [], [], [], []
        start = time.time()

        costs = []
        expert_losses, gpu_losses, rewards,act_losses, critic_loss = [], [], [], [],[]
        for batch_idx, batch in enumerate(data_loader):
            env = Simulate_Env(configs.n_moe_layer, configs.n_e, configs.n_g)
            data = batch.numpy()
            #每个batch有batchsize个样本，每个样本有SampleCnt个经验，每个经验有StateCnt个阶段
            # env.reset函数
            expert_links, expert_nodes, gpu_links, gpu_nodes, mask_expert, mask_gpu = env.reset(data[batch_idx,:,:,:],simu_tokens,configs.expert_size
                                                                                                ,configs.expert_gradsize,configs.token_size)
            act_log_prob = []
            expert_log_prob = []
            gpu_log_prob = []
            env_rewards = []
            env_done = [] # 若所有GPU负载已满，则结束动态调度?

            gpu_select = []
            pool = None
            ep_rewards = - env.initQuality
            
            for state in range(1,StateCnt):
                data1 = Simulate_Dataset(n_e_per_layer,configs.n_e, configs.n_g,configs.n_moe_layer, simu_tokens,1,1, configs.num_ins, 200)
                data1 = data1[0,:,:,:] #生成后继状态
                env_expert_links = deepcopy(torch.Tensor(expert_links)).to(device) # torch.Size([n_e, n_e])
                env_expert_nodes = deepcopy(torch.Tensor(expert_nodes)).to(device) # torch.Size([n_e, fea_dim = 2])
                env_gpu_links = deepcopy(torch.Tensor(gpu_links)).to(device) # torch.Size([n_g, n_g])
                env_gpu_nodes = deepcopy(torch.Tensor(gpu_nodes)).to(device) # torch.Size([n_g, fea_dim = 3])

                env_mask_expert = torch.from_numpy(np.copy(mask_expert)).to(device)
                env_mask_gpu = torch.from_numpy(np.copy(mask_gpu)).to(device)
                expert_prob,expert_index= ppo.old_expert_policy(ep_nodes=env_expert_nodes,
                                                                            ep_links=env_expert_links,
                                                                            gp_nodes=env_gpu_nodes, 
                                                                            gp_links=env_gpu_links,
                                                                            mask_ep=env_mask_expert,
                                                                            mask_gp=env_mask_gpu,
                                                                            old_policy = True)

                gpu_prob,gpu_index,act_prob,act_index = ppo.old_gpu_policy(ep_nodes=env_expert_nodes,
                                                                            ep_links=env_expert_links,
                                                                            gp_nodes=env_gpu_nodes, 
                                                                            gp_links=env_gpu_links,
                                                                            mask_ep=env_mask_expert,
                                                                            mask_gp=env_mask_gpu,
                                                                            ep_index=expert_index,
                                                                            old_policy = True)

                # 记录过程数据
                memory.expert_selection.append(expert_index)
                memory.gpu_selection.append(gpu_index)
                memory.act_selection.append(act_index)
                memory.expert_node_fea.append(env_expert_nodes)
                memory.expert_link_fea.append(env_expert_links)
                expert_log_prob.append(torch.log(expert_prob+1e-10))
                
                memory.gpu_node_fea.append(env_gpu_nodes)
                memory.gpu_link_fea.append(env_gpu_links)
                gpu_log_prob.append(torch.log(gpu_prob+1e-10))
                act_log_prob.append(torch.log(act_prob+1e-10))
                memory.mask_expert.append(env_mask_expert)
                memory.mask_gpu.append(env_mask_gpu)
                # 向环境提交选择的动作和机器，接收新的状态、奖励和完成标志等信息
                expert_nodes, expert_links, gpu_nodes, gpu_links, mask_expert, mask_gpu, dur_time, reward = env.step(expert_index.cpu().numpy(),
                                                                                                gpu_index.cpu().numpy(),
                                                                                                act_index.cpu().numpy(),
                                                                                                data1)
                ep_rewards += reward
                done_list = [0 for _ in range(SampleCnt)]
                done =[1 for _ in range(SampleCnt)]
                env_rewards.append(deepcopy(reward))
                env_done.append(done_list if state != StateCnt-1 else done)

            memory.expert_logprobs.append(expert_log_prob)
            memory.gpu_logprobs.append(gpu_log_prob)
            memory.act_logprobs.append(act_log_prob)
            memory.env_rewards.append(torch.tensor(env_rewards).permute(1, 0))
            memory.env_done.append(torch.tensor(env_done).permute(1, 0))

            # rewards
            ep_rewards -= env.posRewards
            # ppo.update
            torch.autograd.set_detect_anomaly(True)
            expert_loss, gpu_loss, act_loss,value_loss = ppo.update(memory, batch_idx)

            memory.clear_memory()
            mean_time = np.mean(ep_rewards)
            log.append([batch_idx, mean_time])

            # 定期日志记录
            if batch_idx %  10== 0:
                file_writing_obj = open(
                    './' + 'log_' + str(configs.n_e) + '_' + str(configs.n_g) + '_' + str(configs.low) + '_' + str(
                        configs.high) + '.txt', 'w')
                file_writing_obj.write(str(log))

            rewards.append(np.mean(ep_rewards).item())
            expert_losses.append(expert_loss)
            gpu_losses.append(gpu_loss)
            act_losses.append(act_loss)
            critic_loss.append(value_loss)

            cost = time.time() - start
            costs.append(cost)
            step = 5

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

                torch.save(ppo.expert_policy.state_dict(), expert_savePath)
                torch.save(ppo.gpu_policy.state_dict(), gpu_savePath)

                print('  Batch %d/%d, mean_time: %2.3f, expert_loss: %2.4f, gpu_loss: %2.4f,act_loss:%2.4f,critic_loss:%2.4f,took: %2.4fs' %
                      (batch_idx, len(data_loader), mean_time, expert_mean_loss, gpu_mean_loss,act_loss, critic_losss,
                       times[-1]))

                # 性能评估与验证，用于实时监控模型的学习进度和性能
                t4 = time.time()
                validation_log = validate(valid_loader, configs.batch_size, ppo.expert_policy, ppo.gpu_policy).mean()
                if validation_log<record: # 保存最佳模型
                    epoch_dir = os.path.join(filepath, 'best_value100')
                    if not os.path.exists(epoch_dir):
                        os.makedirs(epoch_dir)
                    expert_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_expert'))
                    gpu_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_gpu'))
                    torch.save(ppo.expert_policy.state_dict(), expert_savePath)
                    torch.save(ppo.gpu_policy.state_dict(), gpu_savePath)
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
    main(10)
    total2 = time.time()

    #print(total2 - total1)
