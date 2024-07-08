# RL decision for MoE structure

1.actor-critic framework

an expert actor : chose one epxert every batch
a gpu actor : chose one GPU every batch
an action actor : decide whether to decrease or increase the selected expert replica on the selected gpu

critic : Q-value,calculate the state value

state: an end-to-end latency executed by one moe layer

2.envrionment simulation

token trace:

data i,j represents the number of expert token i transmitted to,on the jth moe layer,shape:(token_number,layer_number)

expert_node: 1:current tokens processed by this expert 2:previous tokens processed by this expert ,use discount methods

shape:(expert_number,2)

expert_links_i_j: 

1:current tokens transmitted from expert_i on the previous layer to expert_j on the subsequent layer

shape:(expert_num//moe_layer_num,expert_num//moe_layer_num,1)

gpu_nodes: 
1: compute speed
2: gpu ultilization
3: gpu memory

shape:(gpu_number,3)
gpu_linksi_j:
1.gpu bandwidth
2.tokens sent from gpu_i to gpu_j
shape:(gpu_number,gpu_number,2)

PS：
1.fastermoe可以取到local和global的gating结果：

(gpu_initial_rank,token_num,gpu_rank):

gate[token_num]=expert_idx:

gpu rank = expert idx / num expert

2.RL修改为一层做一次决策

3.inference

4.显存估计