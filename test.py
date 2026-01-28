
from pufferlib.ocean.threes import Threes                                                                                                         
import numpy as np                                                                                                                                
env = Threes(num_envs=1)                                                                                                                          
obs, _ = env.reset()                                                                                                                              
print('Obs shape:', obs.shape)                                                                                                                    
print('Obs dtype:', obs.dtype)                                                                                                                    
print('Initial obs:', obs[0])                                                                                                                     
for _ in range(10):                                                                                                                               
    obs, rew, term, trunc, info = env.step([np.random.randint(4)])                                                                                
    print(f'Rew: {rew[0]:.4f}, Term: {term[0]}, Obs[:16]: {obs[0, :16]}')                                                                         
