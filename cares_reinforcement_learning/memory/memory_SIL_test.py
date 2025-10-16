# First version under development of self immitation

import numpy as np
import torch
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory 

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.configurations import AlgorithmConfig


from cares_reinforcement_learning.memory.SIL import SelfImitation


# get config; get experiences; add positive reward; test

def test_memory_sil():
            
            alg_config = 'PPO_SIL' gamma=0.99 G=1 G_model=1 number_steps_per_train_policy=5000 buffer_size=1000000 batch_size=256 max_steps_exploration=0 max_steps_training=10000000 image_observation=0 model_path=None actor_lr=0.0001 critic_lr=0.001 eps_clip=0.2 updates_per_iteration=10 actor_config=MLPConfig(layers=[TrainableLayer(layer_category='trainable', layer_type='Linear', in_features=None, out_features=1024, params={}), FunctionLayer(layer_category='function', layer_type='ReLU', params={}), TrainableLayer(layer_category='trainable', layer_type='Linear', in_features=1024, out_features=1024, params={}), FunctionLayer(layer_category='function', layer_type='ReLU', params={}), TrainableLayer(layer_category='trainable', layer_type='Linear', in_features=1024, out_features=None, params={}), FunctionLayer(layer_category='function', layer_type='Tanh', params={})]) critic_config=MLPConfig(layers=[TrainableLayer(layer_category='trainable', layer_type='Linear', in_features=None, out_features=1024, params={}), FunctionLayer(layer_category='function', layer_type='ReLU', params={}), TrainableLayer(layer_category='trainable', layer_type='Linear', in_features=1024, out_features=1024, params={}), FunctionLayer(layer_category='function', layer_type='ReLU', params={}), TrainableLayer(layer_category='trainable', layer_type='Linear', in_features=1024, out_features=1, params={})])
            experiences_0 = [[[-0.07621954 -0.14624923 -0.09004896  0.22143069  0.20273302 -0.54783741 -0.9277934  -0.84035018], 
                             [-0.6877456  0.5441818], 
                             1.0, 
                             [-0.08310095 -0.14586037 -0.10213325  0.20893706 -0.70857481  0.3276722 -0.28115711 -0.38991389], 
                             False], [[-0.08310095 -0.14586037 -0.10213325  0.20893706 -0.70857481  0.3276722 -0.28115711 -0.38991389], 
                                      [-0.83427185 -1.3950421 ], 
                                      0.0, 
                                      [-0.09797222 -0.14831435 -0.10775639  0.19898058 -0.75339996 -0.39502719 -0.28115711 -0.58611389], 
                                      False], [[-0.09797222 -0.14831435 -0.10775639  0.19898058 -0.75339996 -0.39502719 -0.28115711 -0.58611389], 
                                               [-0.8748734  -0.45810604], 
                                               1.0, 
                                               [-0.11317581 -0.15315475 -0.11337954  0.18510011 -0.75258325 -0.14512472 -0.28115711 -0.78231389], 
                                               True]]
            
            memory_factory = MemoryFactory()
            memory_SIL = memory_factory.create_memory(alg_config)  ##add a memory for SIL
            print('created memory_SIL')

            ###sil Instantion
            sil = SelfImitation(memory_SIL)
            print('sil instantion')

            for i in range(3):
                #input test date
                states, actions, rewards, next_states, dones = experiences_0

                sil.setp(states, actions, rewards, next_states, dones)

                #output test data
                good_experiences = memory_SIL.flush()

                print('saved good trajectory number:', len(good_experiences[2]))
                print('prositive reward:')
                print(good_experiences)







    

        
