# MA-PPO for RL

## why PPO

PPO is a classic reinforcement learning (RL) algorithm based on the actor-critic framework, characterized by its ability to save training data, thereby improving training efficiency. Additionally, this algorithm achieves good results in tasks involving multiple agents as well as in scenarios related to machine learning systems (mlsys).

## work flow




![ppo](img\516cc8aefe80ab9d548aa68f006dbf8.png)




the detailed explanation of the formula to calculate advantages is:


![adv](img\65adb4a0d41ed84391f89242ad9854a.png)


while the calulation of Q is discount reward and V is simulated by the critic 