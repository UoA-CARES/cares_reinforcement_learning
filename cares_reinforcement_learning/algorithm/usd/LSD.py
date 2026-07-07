from dataclasses import replace
import logging
import os
from typing import Any

import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F

from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import LSDConfig
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory import memory_sampler
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.LSD.encoder import Encoder
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation, SARLObservationTensors

import tracemalloc
import gc


class LSD(SARLAlgorithm[np.ndarray]):
    def __init__(
        self,
        skills_agent: SAC,
        encoder: Encoder,
        config: LSDConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="usd", config=config, device=device)
        
        # tracemalloc.start(25)
        # self._last_snapshot = None

        self.skills_agent = skills_agent
        self.encoder_net = encoder.to(device)

        # TODO: other ones require a num_skills field to evaluate properly, check @property func below. Need to fix
        self.skill_dim = config.skill_dim # num of discrete skills or dimensions for continuouse skill
        
        self.is_discrete = config.is_discrete

        # training loop control params from config
        self.batch_size=config.batch_size
        self.minibatch_size = config.minibatch_size

        self.agent_loops_per_batch = config.agent_loops_per_batch
        self.agent_num_minibatches_per_batch = config.agent_num_minibatches_per_batch

        self.encoder_loops_per_batch = config.encoder_loops_per_batch
        self.encoder_num_minibatches_per_batch = config.encoder_num_minibatches_per_batch

        self.is_using_buffer = config.is_using_buffer

        # set skill prior
        if self.is_discrete:
            self.p_z = np.full(
                self.skill_dim, 1.0 / self.skill_dim, dtype=np.float32
            )  # (K,)
        else:
            self.p_z = None  # ignore as using standard normal built into np

        # sample first skill
        self._sample_new_z()

        # create state encoder optimiser
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder_net.parameters(), lr=config.encoder_lr
        )

    # TODO: fix this. This is here to make sure the current eval run work.
    @property
    def num_skills(self):
        if self.is_discrete:
            return self.skill_dim
        else:
            raise NotImplementedError("Evaluation for continuous skills not implemented yet.")
        
    # TODO: fix this too
    def set_skill(self, skill: int, evaluation: bool = False) -> None:
        if not self.is_discrete:
            raise NotImplementedError("Evaluation for continuous skills not implemented yet.")
        if skill < 0 or skill >= self.num_skills:
            raise ValueError(f"Skill index {skill} is out of bounds.")

        self.z = skill
    
    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:
        observation = self._normalize_obs(observation)
        observation = replace(
            observation,
            vector_state=self._concat_state_and_skill(observation.vector_state),
        )

        action_sample = self.skills_agent.act(observation, evaluation)
        action_sample.extras["skill"] = self.z
        return action_sample

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}


        # Original paper training loop:
        # Collect samples for training "epoch":
        #   - for "max_optimization_epoches"(param):
        #       - for "trans_optimization_epochs" times: 
        #           - Draw with replacement "trans_minibatch_size" transitions (just sample from replay buffer if using buffer)
        #           - Update SAC (* CSD and LSD implementation differs)
        #   - for "te_max_optimization_epochs"(param):
        #       - for "te_trans_optimization_epochs" times: 
        #           - Draw with replacement "te_trans_minibatch_size" transitions (just sample from replay buffer if using buffer)
        #           - Update encoder
        #   - soft update q net (SAC q net) (in ORIGINAL, in CSD implementation omit since target update in update sac step)

        # skip if there are not enough samples in mem buffer
        if len(memory_buffer) < self.batch_size:
            return {}

        ########################################################
        # update agent
        for _ in range(self.agent_loops_per_batch):
            # ^ in original training only runs per epoch for certain rounds
            for _ in range(self.agent_num_minibatches_per_batch):
        
                sample_tensor,_ = memory_sampler.sample(
                    memory=memory_buffer,
                    batch_size=self.minibatch_size,
                    device=self.device,
                    use_per_buffer=0,  # DIAYN does not use PER
                )

                observation_tensor = sample_tensor.observation
                actions_tensor = sample_tensor.action
                rewards_tensor = sample_tensor.reward
                next_observation_tensor = sample_tensor.next_observation
                dones_tensor = sample_tensor.done
                weights_tensor = sample_tensor.weights
                train_data = sample_tensor.train_data

                skill_tensor = self._get_skill_tensor(train_data, observation_tensor.vector_state.dtype)

                # compute reward for RL agent to optimize----------------------------
                rewards_tensor = self.compute_representation_skill_prod(
                    observation_tensor.vector_state,
                    next_observation_tensor.vector_state,
                    skill_tensor
                ).detach().unsqueeze(-1) #(B,) -> (B,1), needed for some reason

                # Create SARLObservation obj with skill concated to state ---------------------------------------------------------
                states_z_tensor = torch.cat(
                    [observation_tensor.vector_state, skill_tensor], dim=1
                )
                next_states_z_tensor = torch.cat(
                    [next_observation_tensor.vector_state, skill_tensor], dim=1
                )
                
                observation_z_tensor = replace(
                        observation_tensor,
                        vector_state=states_z_tensor,
                    )
                next_observation_z_tensor = replace(
                    next_observation_tensor,
                    vector_state=next_states_z_tensor,
                )

                # Update SAC agent ---------------------------------------------------------
                agent_info, _ = self.skills_agent.update_from_batch(
                        observation_tensor=observation_z_tensor,
                        actions_tensor=actions_tensor,
                        rewards_tensor=rewards_tensor,
                        next_observation_tensor=next_observation_z_tensor,
                        dones_tensor=dones_tensor,
                        weights_tensor=weights_tensor
                    )
                info |= agent_info

        ##########################################################################
        # update state encoder
        for _ in range(self.encoder_loops_per_batch):
            for _ in range(self.encoder_num_minibatches_per_batch):
                sample_tensor,_ = memory_sampler.sample(
                    memory=memory_buffer,
                    batch_size=self.minibatch_size,
                    device=self.device,
                    use_per_buffer=0,  # DIAYN does not use PER
                )

                observation_tensor = sample_tensor.observation
                actions_tensor = sample_tensor.action
                rewards_tensor = sample_tensor.reward
                next_observation_tensor = sample_tensor.next_observation
                dones_tensor = sample_tensor.done
                weights_tensor = sample_tensor.weights
                train_data = sample_tensor.train_data

                skill_tensor = self._get_skill_tensor(train_data, observation_tensor.vector_state.dtype)

                representation_skill_prod = self.compute_representation_skill_prod(
                    observation_tensor.vector_state,
                    next_observation_tensor.vector_state,
                    skill_tensor
                ) #(B,)

                # Update state encoder
                representation_skill_prod_mean = representation_skill_prod.mean() #()
                encoder_loss = -representation_skill_prod_mean

                self.encoder_optimizer.zero_grad()
                encoder_loss.backward()
                self.encoder_optimizer.step()

                info |= {"encoder loss":encoder_loss.mean().item()}
        
        if not self.is_using_buffer:
            memory_buffer.clear()
            
            # snapshot = tracemalloc.take_snapshot()
            # if self._last_snapshot is not None:
            #     top_stats = snapshot.compare_to(self._last_snapshot, "lineno")
            #     print("=== Top 10 growing allocations ===")
            #     for stat in top_stats[:10]:
            #         print(stat)
            # self._last_snapshot = snapshot
            
            # print("bru====================================================")
            # # found = 0
            # for obj in gc.get_objects():
            #     if torch.is_tensor(obj):
            #         tb = tracemalloc.get_object_traceback(obj)
            #         if tb and any(
            #             frame.filename.endswith("SAC.py") 
            #             for frame in tb
            #         ):
            #             found += 1
            #             print(f"--- leaked tensor #{found}: shape={tuple(obj.shape)}, "
            #                 f"requires_grad={obj.requires_grad}, grad_fn={obj.grad_fn}")
            #             for ref in gc.get_referrers(obj):
            #                 print("  held by:", type(ref), repr(ref)[:200])
            #             if found >= 5:
            #                 break

        return info

    def _get_skill_tensor(self,train_data:dict,dtype:torch.dtype):
        '''Use train_data from sample to create (B,skill_dim) tensor'''
        # Extract skills from the sampled transitions into tensor-------------------------
        skills = [extra["skill"] for extra in train_data]
        if self.is_discrete:
            z_code_tensor= torch.tensor(skills, dtype=torch.long, device=self.device) #(B,) each entry is the number for the skill
            # Concatenate zs (skills) as one-hot to states
            # pylint: disable-next=not-callable
            skill_tensor = F.one_hot(z_code_tensor, num_classes=self.skill_dim).to(dtype) # (B,skill_dim), one hot for chosen skill
        else:
            skill_tensor = torch.tensor(skills) #(B,skill_dim)
        
        return skill_tensor
    
    def compute_representation_skill_prod(
            self,
            state_tensor: torch.Tensor, #(B, state size)
            next_state_tensor: torch.Tensor, #(B, state size)
            skill_tensor: torch.Tensor #(B, skill_dim)
        ):
        '''
        Calculate (phi(s')-phi(s))z. 
        
        Think def of dot prod: length of state diff vec (in embedding space) projected onto z, times length of z. High reward
        implies: good alighment, ALSO scale with length of embedding vec. 
        
        Lipschitz constraint: large diff in states embedding implies large diff in actual state space. i.e. if s and s' is the
        same, encoder can't arbitrarily make phi(s') huge thus maximising this reward.
        '''
        current_z = self.encoder_net(state_tensor) #(B, skill_dim)
        next_z = self.encoder_net(next_state_tensor) #(B, skill_dim)
        delta_z:torch.Tensor = next_z - current_z #(B, skill_dim)

        if self.is_discrete:
            # (B,1,skill_dim)
            delta_z = delta_z.reshape(delta_z.size(0), 1, delta_z.size(1))
            # (B,skill_dim,skill_dim). Batches of one hot encoding for each skill:
            # e.g. [[1,0,0],[0,1,0],[0,0,1],...]
            eye_z = (
                torch.eye(self.skill_dim, device=self.device)
                .reshape(1, self.skill_dim, self.skill_dim)
                .expand(delta_z.size(0), -1, -1)
            )
            # (B,skill_dim), effectively: each option gets a score of how much latent change align with that dimension
            logits = (eye_z * delta_z).sum(dim=2)

            # (B,skill_dim), onehot for chosen skill -> centered around 0 and scaled
            masks = (
                (skill_tensor - skill_tensor.mean(dim=1, keepdim=True))
                * (self.skill_dim)
                / (self.skill_dim - 1 if self.skill_dim != 1 else 1)
            )

            return (logits * masks).sum(dim=1)  # (B,)

        else:
            return (delta_z * skill_tensor).sum(dim=1)  # (B,)

    def episode_done(self):
        """
        Sample new skill from p(z) when episode finishes.
        """
        self._sample_new_z()
        return super().episode_done()

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state = replace(
            state, vector_state=self._concat_state_and_skill(state.vector_state)
        )
        return self.skills_agent._calculate_value(state, action)

    ###### ACTUAL CHEATS ##################################################################
    def _normalize_obs(self,obs:SARLObservation|SARLObservationTensors):
        '''!!!!!!!!!! ACTUAL CHEATS. cheetah'''
        # Precomputed mean and std of the state dimensions from 10000 length-50 random rollouts (without early termination)
        # cheetah
        # normalizer_mean_np = np.array(
        #     [-0.07861924, -0.08627162, 0.08968642, 0.00960849, 0.02950368, -0.00948337, 0.01661406, -0.05476654,
        #     -0.04932635, -0.08061652, -0.05205841, 0.04500197, 0.02638421, -0.04570961, 0.03183838, 0.01736591,
        #     0.0091929, -0.0115027])
        # normalizer_std_np = np.array(
        #     [0.4039283, 0.07610687, 0.23817, 0.2515473, 0.2698137, 0.26374814, 0.32229397, 0.2896734, 0.2774097,
        #     0.73060024, 0.77360505, 1.5871304, 5.5405455, 6.7097645, 6.8253727, 6.3142195, 6.417641, 5.9759197])

        # ant
        normalizer_mean_np = np.array(
            [0.00486117, 0.011312, 0.7022248, 0.8454677, -0.00102548, -0.00300276, 0.00311523, -0.00139029,
             0.8607109, -0.00185301, -0.8556998, 0.00343217, -0.8585605, -0.00109082, 0.8558013, 0.00278213,
             0.00618173, -0.02584622, -0.00599026, -0.00379596, 0.00526138, -0.0059213, 0.27686235, 0.00512205,
             -0.27617684, -0.0033233, -0.2766923, 0.00268359, 0.27756855])
        normalizer_std_np = np.array(
            [0.62473416, 0.61958003, 0.1717569, 0.28629342, 0.20020866, 0.20572574, 0.34922406, 0.40098143,
             0.3114514, 0.4024826, 0.31057045, 0.40343934, 0.3110796, 0.40245822, 0.31100526, 0.81786263, 0.8166509,
             0.9870919, 1.7525449, 1.7468817, 1.8596431, 4.502961, 4.4070187, 4.522444, 4.3518476, 4.5105968,
             4.3704205, 4.5175962, 4.3704395])
        
        if isinstance(obs,SARLObservationTensors):
            # print(obs.vector_state.dtype)
            normalizer_mean = torch.tensor(normalizer_mean_np, dtype=torch.float32, device=self.device) 
            normalizer_std = torch.tensor(normalizer_std_np, dtype=torch.float32, device=self.device)
            normalized_vec = (obs.vector_state - normalizer_mean) / (torch.sqrt(normalizer_std) + 1e-8)
            new_obs = replace(obs, vector_state = normalized_vec)
            # print(f"Normalised Tensor Obs: {new_obs}")
            return new_obs
        else:
            normalized_vec = (obs.vector_state - normalizer_mean_np) / (np.sqrt(normalizer_std_np) + 1e-8)
            new_obs = replace(obs, vector_state = normalized_vec)
            # print(f"Normalised np obs: {new_obs}")
            return new_obs
        

    ###### LATENT RELATED UTILS ############################################################
    def _concat_state_and_skill(self, state: np.ndarray) -> np.ndarray:
        if self.is_discrete:
            z_one_hot = np.zeros(self.skill_dim)
            z_one_hot[self.z] = 1
            return np.concatenate([state, z_one_hot])
        else:
            return np.concatenate([state, self.z])

    def _sample_new_z(self):
        """Sample new skill z following p(z)"""
        # NOTE: different z in continuous and discrete case
        #   - Discrete: z is the number for the skill (NOT one-hot)
        #   - Continuous: z is sampled (skill_dim) shaped tensor
        if self.is_discrete:
            self.z = np.random.choice(self.skill_dim, p=self.p_z)
        else:
            self.z = np.random.randn(self.skill_dim)

    ###### SAVING UTIL ####################################################################

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Save skills agent
        self.skills_agent.save_models(filepath, f"{filename}_skill_agent")

        # Save LSD-specific state in a single checkpoint
        checkpoint = {
            "encoder_state_dict": self.encoder_net.state_dict(),
            "encoder_optimizer_state_dict": self.encoder_optimizer.state_dict(),
            "z": self.z,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_lsd.pth")
        logging.info("LSD models and state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.skills_agent.load_models(filepath, f"{filename}_skill_agent")

        checkpoint = torch.load(f"{filepath}/{filename}_diayn.pth")

        self.encoder_net.load_state_dict(checkpoint["encoder_state_dict"])
        self.encoder_optimizer.load_state_dict(
            checkpoint["encoder_optimizer_state_dict"]
        )

        self.z = checkpoint.get("z", self.z)
        logging.info("LSD models and state have been loaded...")
