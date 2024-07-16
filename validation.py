from Params import configs
from copy import deepcopy
from Simulate_Env import Simulate_Env
import copy
from uniform_instance import Simulate_Dataset
from agent_utils import sample_select_action
from agent_utils import greedy_select_action
import numpy as np
import torch
import matplotlib.pyplot as plt
from Params import configs
def validate(vali_set,batch_size, policy_expert,policy_gpu):
    policy_expert = copy.deepcopy(policy_expert)
    policy_gpu = copy.deepcopy(policy_gpu)
    policy_expert.eval()
    policy_gpu.eval()
    def eval_model_bat(bat,i):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()
            simu_tokens = 50
            env = Simulate_Env(configs.n_moe_layer, configs.n_e, configs.n_g)    
            device = torch.device(configs.device)
            expert_links, expert_nodes, gpu_links, gpu_nodes, mask_expert, mask_gpu = env.reset(data[0,:,:,:],simu_tokens,configs.expert_size
                                                                                                ,configs.expert_gradsize,configs.token_size)

            n_e_per_layer = configs.n_e // configs.n_moe_layer
            
            ep_rewards = - env.initQuality
            SampleCnt = configs.batchsize

            StateCnt = 10
            for state in range(1,StateCnt):
                data1 = Simulate_Dataset(n_e_per_layer,configs.n_e, configs.n_g,configs.n_moe_layer, simu_tokens,1,1, configs.num_ins, 200)
                data1 = data1[0,:,:,:] #生成后继状态
                env_expert_links = deepcopy(torch.Tensor(expert_links)).to(device) # torch.Size([n_e, n_e])
                env_expert_nodes = deepcopy(torch.Tensor(expert_nodes)).to(device) # torch.Size([n_e, fea_dim = 2])
                env_gpu_links = deepcopy(torch.Tensor(gpu_links)).to(device) # torch.Size([n_g, n_g])
                env_gpu_nodes = deepcopy(torch.Tensor(gpu_nodes)).to(device) # torch.Size([n_g, fea_dim = 3])

                env_mask_expert = torch.from_numpy(np.copy(mask_expert)).to(device)
                env_mask_gpu = torch.from_numpy(np.copy(mask_gpu)).to(device)
                expert_prob,expert_index= policy_expert(ep_nodes=env_expert_nodes,
                                                                            ep_links=env_expert_links,
                                                                            gp_nodes=env_gpu_nodes, 
                                                                            gp_links=env_gpu_links,
                                                                            mask_ep=env_mask_expert,
                                                                            mask_gp=env_mask_gpu,
                                                                            old_policy = True)

                gpu_prob,gpu_index,act_prob,act_index = policy_gpu(ep_nodes=env_expert_nodes,
                                                                            ep_links=env_expert_links,
                                                                            gp_nodes=env_gpu_nodes, 
                                                                            gp_links=env_gpu_links,
                                                                            mask_ep=env_mask_expert,
                                                                            mask_gp=env_mask_gpu,
                                                                            ep_index=expert_index,
                                                                            old_policy = True)

                # 向环境提交选择的动作和机器，接收新的状态、奖励和完成标志等信息
                expert_nodes, expert_links, gpu_nodes, gpu_links, mask_expert, mask_gpu, dur_time, reward = env.step(expert_index.cpu().numpy(),
                                                                                                gpu_index.cpu().numpy(),
                                                                                                act_index.cpu().numpy(),
                                                                                                data1)
                ep_rewards += reward
                done_list = [0 for _ in range(SampleCnt)]
                done =[1 for _ in range(SampleCnt)]
                
            cost = ep_rewards
            C_max.append(cost)
        return torch.tensor(cost)
    #make_spans.append(rewards - env.posRewards)
    #print(env.mchsStartTimes,env.mchsEndTimes,env.opIDsOnMchs)
    #print('REWARD',rewards - env.posRewards)
    totall_cost = torch.cat([eval_model_bat(bat,i) for i,bat in enumerate(vali_set)], 0)

    return totall_cost



if __name__ == '__main__':

    from uniform_instance import uni_instance_gen,FJSPDataset
    import numpy as np
    import time
    import argparse
    from Params import configs

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=30, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=20, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=30, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=20, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=-99, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Cap seed for validate set generation')
    parser.add_argument('--n_vali', type=int, default=100, help='validation set size')
    params = parser.parse_args()

    N_JOBS_P = params.Pn_j
    N_MACHINES_P = params.Pn_m
    LOW = params.low
    HIGH = params.high
    N_JOBS_N = params.Nn_j
    N_MACHINES_N = params.Nn_m
    from torch.utils.data import DataLoader
    from PPOwithValue import PPO
    import torch
    import os
    from torch.utils.data import Dataset
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_JOBS_P,
              n_m=N_MACHINES_P,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    filepath = 'saved_network'
    filepath = os.path.join(filepath, 'FJSP_J%sM%s' % (30,configs.n_m))
    #filepath = os.path.join(filepath, '%s_%s' % (0,239))
    filepath = os.path.join(filepath, 'best_value0')

    job_path = './{}.pth'.format('policy_job')
    mch_path = './{}.pth'.format('policy_mch')



    '''filepath = 'saved_network'
    filepath = os.path.join(filepath,'%s'%19)
    job_path = './{}.pth'.format('policy_job'+str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
    mch_path = './{}.pth'.format('policy_mch'+ str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))'''

    job_path = os.path.join(filepath,job_path)
    mch_path = os.path.join(filepath, mch_path)

    ppo.policy_job.load_state_dict(torch.load(job_path))
    ppo.policy_mch.load_state_dict(torch.load(mch_path))
    num_val = 10
    batch_size = 1
    SEEDs = [200]
    result = []
    loade = False


    for SEED in SEEDs:

        mean_makespan = []
        #np.random.seed(SEED)
        if loade:
            validat_dataset = np.load(file="FJSP_J%sM%s_unew_test_data.npy" % (configs.n_j, configs.n_m))
            print(validat_dataset.shape[0])
        else:
            validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, num_val, SEED)
        valid_loader = DataLoader(validat_dataset, batch_size=batch_size)
        vali_result = validate(valid_loader,batch_size, ppo.policy_job, ppo.policy_mch)
        #mean_makespan.append(vali_result)
        print(vali_result,np.array(vali_result).mean())

    # print(min(result))

