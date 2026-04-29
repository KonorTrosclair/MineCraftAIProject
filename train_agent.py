import os
import shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
import torch.nn.functional as F
from minestudio.models.vpt import body


import numpy as np
from torch.distributions import Categorical
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import HardResetCallback, CommandsCallback, RecordCallback
from minestudio.models import VPTPolicy
from minestudio.models.base_policy import dict_map, recursive_tensor_op


# CONFIG
LEARNING_RATE  = 3e-7
NUM_EPISODES   = 200
MAX_STEPS      = 5000
GAMMA          = 0.99
SAVE_EVERY     = 5
UPDATE_EVERY   = 200  # update policy every 200 steps (this is due to memory limitations) However to avoid policy dillution we only update if there is a reward more than 0
MEMORY_RESET_EVERY = 200  # reset memory every 200 steps (not used anymore)
seed=SET_SEED = 92641161
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./output", exist_ok=True)
os.makedirs("./best_output", exist_ok=True)


# LOAD MODEL
policy = VPTPolicy.from_pretrained(
    "CraftJarvis/MineStudio_VPT.foundation_model_1x" # model that is pretrained on basic behavior cloning
).to(DEVICE)

checkpoint = torch.load('./checkpoints/policy_best.pt') # loads from best policy can be changed if a different checkpoint is desired

if isinstance(checkpoint, dict) and 'policy' in checkpoint:
    policy.load_state_dict(checkpoint['policy'])
    best_reward = checkpoint['best_reward']
else:
    policy.load_state_dict(checkpoint)
    best_reward = 312.00  

print(f"Loaded checkpoint, best reward: {best_reward:.2f}")
policy.train()


# fake_obs = {"image": np.zeros((128, 128, 3), dtype=np.uint8)}
# input_batched = dict_map(policy._batchify, fake_obs)
# print("Testing patched forward...")
# latents, state_out = policy.forward(input_batched, None)
# print("Patch working! keys:", latents['pi_logits'].keys())

optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)


# REWARD TRACKER
LOG_REWARD = 1
PLANK_REWARD = 500
CRAFT_TABLE_REWARD = 10000
PICK_REWARD = 1500
STONE_REWARD = 8
COAL_REWARD = 1000

class RewardTracker:
    def __init__(self):
        self.menu_timer = 0
        self.step_count = 0
        self.movement_threshold = 0.65
        self.prev_total_progress = 0
        self.prev_x = 0
        self.prev_z = 0
        self.max_logs   = 0
        self.max_planks = 0
        self.max_sticks = 0
        self.max_stone  = 0
        self.max_coal   = 0
        self.rewarded_crafting_table = False
        self.rewarded_pickaxe = False
        self.first_inv_open = False
        self.has_opened_inventory = False

    #gets num of item in inventory
    def get_inventory_count(self, info, item_name): 
        total = 0
        for slot in info['inventory'].values():
            if item_name in slot['type']: 
                total += slot['quantity']
        return total

    #function dedicated to penalizing the aigent for hoarding and to reward if agent reached lifetime high
    def process_item_reward(self, name, current, max_ever, cap, reward_val, hoard_limit, penalty_val): 
        delta_reward = 0.0
        
        if current > max_ever:
            print(f"  *** {name.upper()} NEW RECORD: {max_ever} -> {current} ***")
            
            items_to_reward = max(0, min(current, cap) - max_ever)
            delta_reward += items_to_reward * reward_val
            
            if current > hoard_limit:
                new_excess = current - max(max_ever, hoard_limit)
                if new_excess > 0:
                    delta_reward -= new_excess * penalty_val
                    print(f"      [Penalty] Hoarding {name}: -{new_excess * penalty_val:.2f}")
        
        return delta_reward

    #computes rewards
    def compute(self, obs, info):
        reward = 0.0
        self.step_count += 1


        
        
        if info.get('is_attacking', False) and 'log' in info.get('looking_at_block', ''):
            reward += 0.05 

        # LOGS 
        current_logs = self.get_inventory_count(info, 'log')
        reward += self.process_item_reward('log', current_logs, self.max_logs, 10, LOG_REWARD, 12, 4)
        self.max_logs = max(self.max_logs, current_logs)

        # PLANKS 
        current_planks = self.get_inventory_count(info, 'planks')
        reward += self.process_item_reward('planks', current_planks, self.max_planks, 40, PLANK_REWARD, 48, 2)
        self.max_planks = max(self.max_planks, current_planks)

        # STICKS 
        current_sticks = self.get_inventory_count(info, 'stick')
        reward += self.process_item_reward('stick', current_sticks, self.max_sticks, 4, 2.0, 12, 0.1)
        self.max_sticks = max(self.max_sticks, current_sticks)

        # STONE
        current_stone = self.get_inventory_count(info, 'cobblestone')
        reward += self.process_item_reward('stone', current_stone, self.max_stone, 32, STONE_REWARD, 64, 0.1)
        self.max_stone = max(self.max_stone, current_stone)

        # COAL
        current_coal = self.get_inventory_count(info, 'coal')
        reward += self.process_item_reward('coal', current_coal, self.max_coal, 8, COAL_REWARD, 16, 0.5)
        self.max_coal = max(self.max_coal, current_coal)

        # CRAFTING TABLE
        if not self.rewarded_crafting_table and self.get_inventory_count(info, 'crafting_table') > 0:
            reward += CRAFT_TABLE_REWARD
            self.rewarded_crafting_table = True
            print(f"  *** CRAFTING TABLE OBTAINED ***")

        # WOOD PICKAXE
        if not self.rewarded_pickaxe and self.get_inventory_count(info, 'wooden_pickaxe') > 0:
            reward += PICK_REWARD
            self.rewarded_pickaxe = True
            print(f"  *** PICKAXE OBTAINED ***")
        
        
    
        # reward -= 0.01

        if info.get('health', 20) <= 0:
            reward -= 10.0

        return reward


# RETURNS
def compute_returns(rewards, gamma=GAMMA):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
    if returns.std() > 0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# ACTION + LOGPROB
def get_action_and_logprob(policy, obs, memory):
    input_batched = dict_map(policy._batchify, obs)
    
    if memory is not None:
        memory_in = recursive_tensor_op(lambda x: x.unsqueeze(0), memory)
    else:
        memory_in = None

    latents, state_out = policy.forward(input_batched, memory_in)

    action = policy.pi_head.sample(latents['pi_logits'], deterministic=False)
    pi_logits = latents['pi_logits']

    button_action = action['buttons'].squeeze()
    camera_action = action['camera'].squeeze()

    button_dist = Categorical(logits=pi_logits['buttons'].squeeze())
    camera_dist = Categorical(logits=pi_logits['camera'].squeeze())

    log_prob = button_dist.log_prob(button_action) + camera_dist.log_prob(camera_action)

    state_out = recursive_tensor_op(lambda x: x[0], state_out)
    state_out = recursive_tensor_op(lambda x: x.detach(), state_out)

    return action, state_out, log_prob


# ENVIRONMENT
# went for a set seed and tp to coords that are always in a forest
# This is to enssure policies do not dillute in situations where there are no trees present.
# and same forest each run makes training a little simpler due to familiar enviorment
HOME_X, HOME_Y, HOME_Z = 4674.5, 63.0, -1573.5
IX, IY, IZ = int(HOME_X), int(HOME_Y), int(HOME_Z)

FIXED_SPAWN = [{
    "seed": 92641161,
    "position": [4674.5, -1573.5, 63.0]
}]

env = MinecraftSim(
    obs_size=(128, 128),
    seed=SET_SEED, 
    callbacks=[
        HardResetCallback(spawn_positions=FIXED_SPAWN),
        CommandsCallback(commands=[
            f'/setworldspawn {IX} {IY} {IZ}',
            f'/spawnpoint @a {IX} {IY} {IZ}',
            f'/tp @a {HOME_X} {HOME_Y} {HOME_Z}',
            f"/give @p oak_log 10",
        ]),
        RecordCallback(record_path="./output", fps=30, frame_type="pov")
    ]
)

def detach_memory(memory):
    if memory is None:
        return None
    return recursive_tensor_op(lambda x: x.detach().cpu(), memory)

def to_device_memory(memory, device):
    if memory is None:
        return None
    return recursive_tensor_op(lambda x: x.to(device), memory)


# TRAIN LOOP

import random
print(f"Training on {DEVICE}")

for episode in range(NUM_EPISODES):
    # current_seed = random.randint(1, 1000000)
    
    # AUTO RESTART ON CRASH -- MC enviorment seems to crash after every 8 episodes 
    # this ensures that the minecraft enviroment will restart to continue training
    try:
        obs, info = env.reset()

        
    except Exception as e:
        print(f"Environment crashed: {e}. Restarting...")
        try:
            env.close()
        except:
            pass
        

        env = MinecraftSim(
            obs_size=(128, 128),
            seed=SET_SEED, 
            callbacks=[
                HardResetCallback(spawn_positions=FIXED_SPAWN),
                CommandsCallback(commands=[
                    
                    f'/setworldspawn {IX} {IY} {IZ}',
                    f'/spawnpoint @a {IX} {IY} {IZ}',
                    f'/tp @a {HOME_X} {HOME_Y} {HOME_Z}',
                    f"/give @p oak_log 10",
                ]),
                RecordCallback(record_path="./output", fps=30, frame_type="pov")
            ]
        )
        obs, info = env.reset()

    memory = None
    rewards   = []
    log_probs = []
    total_reward = 0.0
    last_loss = 0.0
    reward_tracker = RewardTracker()
    print(f"Tracker reset: max_logs={reward_tracker.max_logs}")

    for step in range(MAX_STEPS):
        try:
            action, memory, log_prob = get_action_and_logprob(policy, obs, to_device_memory(memory, DEVICE))
            memory = detach_memory(memory)
            obs, _, terminated, truncated, info = env.step(action)
            # if step == 0:
            #     print("Info keys:", info.keys())
            #     print("Info sample:", info)
        except Exception as e:
            print(f"Step failed: {e}. Ending episode early.")
            break

        reward = reward_tracker.compute(obs, info)
        total_reward += reward

        # if step % 500 == 0:
        #     print(f"    > Step {step:4d} | Current Reward: {total_reward:6.2f} | Inventory: Logs({reward_tracker.max_logs}) Planks({reward_tracker.max_planks})")
        
        rewards.append(reward)
        log_probs.append(log_prob)

        # if step % MEMORY_RESET_EVERY == 0: not used since 
        #     memory = detach_memory(memory)

        # must update the policy every 200 steps instead of after an episode due to vram memory constraints
        # to manage the memory we update the policy so the vram is not holding on to all the policy changes of the entire episode
        if len(rewards) >= UPDATE_EVERY: 
            # Ensures that the policy only updates after 200 episodes where the net rewared is greater than 0
            # this avoids dilluting the policy to some extent
            if any(abs(r) > 0.01 for r in rewards): 
                returns = compute_returns(rewards)
                stacked = torch.stack(log_probs)
                loss = -(stacked * returns).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()
                last_loss = loss.item()
                print(f"  [Update @ step {step}] Loss: {last_loss:.6f}")
            else:
                print(f"  [Skipped @ step {step}] No reward signal")
            
            rewards = []
            log_probs = []
            memory = None
            torch.cuda.empty_cache()

        if terminated or truncated:
            break

    print(
        f"Episode {episode+1}/{NUM_EPISODES} | "
        f"Steps {step+1} | "
        f"Reward {total_reward:.2f} | "
        f"Loss {last_loss:.4f}"
    )

    # SAVE BEST
    # saves the best checkpoint so we can go back to it whenever a new reward record is reached.
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save({
            'policy': policy.state_dict(),
            'best_reward': best_reward,
            'episode': episode + 1
        }, './checkpoints/policy_best.pt')

        current_video = f"./output/episode_{episode+1}.mp4"
        if os.path.exists(current_video):
            shutil.copy(current_video, "./output/best_episode.mp4")
            
        print(f"  New best! Reward {best_reward:.2f} saved to policy_best.pt")

    if (episode + 1) % SAVE_EVERY == 0:
        path = f"./checkpoints/policy_ep{episode+1}.pt"
        torch.save(policy.state_dict(), path)
        print(f"Saved: {path}")

env.close()
print("Training complete.")