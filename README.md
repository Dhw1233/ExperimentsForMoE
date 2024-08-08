# MA-PPO for RL

## why PPO

PPO is a classic reinforcement learning (RL) algorithm based on the actor-critic framework, characterized by its ability to save training data, thereby improving training efficiency. Additionally, this algorithm achieves good results in tasks involving multiple agents as well as in scenarios related to machine learning systems (mlsys).

## work flow




![ppo](img\516cc8aefe80ab9d548aa68f006dbf8.png)




the detailed explanation of the formula to calculate advantages is:


![adv](img\65adb4a0d41ed84391f89242ad9854a.png)


while the calulation of Q is discount reward and V is simulated by the critic 

## major work flow

### Algorithm 1:PPO algorithm

```plaintext
FOR each episode DO

    initialize trajectory pool D = {}
    FOR j = 1 TO BATCH_SIZE DO
        receive first MoE layer state: s0
        s = s0
        WHILE s is not the last MoE layer state 
            FOR each agent i DO
                generate π(a|s; θ')
                acording to π(a|s; θ'), chose action a_i
                apend a_i to Action set A
            env update(A)
            observer reward r and next MoE layer state s'
            store (s, A, r, s') into trajetory pool D_j
            s = s'
    END FOR

    FOR k = 1 TO K DO  // K is a hyper parameter of training epoches
        FOR each trajectory batch D_j from D DO
            intialize gradient
            FOR s and a in each timestep :
                calculate π(a|s; θ) for each agent
                caclulate Advantage: Adv(S, A; V(s), V_target(s))

                calculate PPO objective：
                ratio = π(a|s; θ) / π(a|s; θ')
                surr1 = ratio * Adv
                surr2 = clip(ratio, 1 - ε, 1 + ε) * Adv

                caclulate loss function L：
                L += - min(surr1, surr2) - λ * V(s) + λ * V_target(s)
                
            caculate L_critic
        END FOR
        update gradient of θ and V(s) based on L and L_critic
        θ' = θ 
    END FOR
END FOR

### Algorithm 2: structure of expert actor and gpu actor

```plaintext
EXPERT_ACTOR:

For state encoder , we use a gnn to embed layer-wise token load ratio (and expert affinity),and a decoder(MLP) to generate the chosen expert
of this layer 

GPU_ACTOR:

For state encoder , we use a gnn to embed gpu cluster feature and inter-gpu traffic of each layer , and it has (3*num_expert) nodes
,for node_{i} it represents current expert replacement on gpu i,for node_{i + gpu_num} it represents add expert replica on gpu_{i},while node_{i+num_gpu*2} for delete

For decoder ,we use the embedding output of the gnn and a decoder(MLP) to generate an action distribution 

### Algorithm 3: fake data generation 

### Algorithm 4: env update
initializaiton: current layer 's expert token ratio, gpu features ,all to all communication and thus a replica number on  it
env update : generate next layer's feature based on the token trace log.

token trace log :
a two-dimension np array:
(token_num,layer_num) is the expert number routed to of this token on this layer. 