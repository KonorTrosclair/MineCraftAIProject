import os
import shutil
import glob
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# PATCH MUST BE ABSOLUTE FIRST
import torch
import torch.nn.functional as F
from minestudio.models.vpt import body

# ALL IMPORTS AFTER PATCH
import numpy as np
from torch.distributions import Categorical
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import HardResetCallback, CommandsCallback, RecordCallback
from minestudio.models import VPTPolicy
from minestudio.models.base_policy import dict_map, recursive_tensor_op
from Fixed_Reset_Callback import FixedResetCallback

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LEARNING_RATE  = 1e-6
NUM_EPISODES   = 200
MAX_STEPS      = 5000
GAMMA          = 0.99
SAVE_EVERY     = 5
UPDATE_EVERY   = 128  # ← update policy every 64 steps instead of full episode
MEMORY_RESET_EVERY = 128  # reset memory every 64 steps
seed=SET_SEED = 92641161
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./output", exist_ok=True)
os.makedirs("./best_output", exist_ok=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
policy = VPTPolicy.from_pretrained(
    "CraftJarvis/MineStudio_VPT.foundation_model_1x"
).to(DEVICE)

checkpoint = torch.load('./checkpoints/policy_best.pt')

# Try new format first, fall back to old format
if isinstance(checkpoint, dict) and 'policy' in checkpoint:
    policy.load_state_dict(checkpoint['policy'])
    best_reward = checkpoint['best_reward']
else:
    # Old format - just the state dict directly
    policy.load_state_dict(checkpoint)
    best_reward = 258.00  # ← set this to whatever your best reward was
# best_reward = -30.0    
print(f"Loaded checkpoint, best reward: {best_reward:.2f}")
policy.train()

# ── TEST PATCH BEFORE MINECRAFT LAUNCHES ──
fake_obs = {"image": np.zeros((128, 128, 3), dtype=np.uint8)}
input_batched = dict_map(policy._batchify, fake_obs)
print("Testing patched forward...")
latents, state_out = policy.forward(input_batched, None)
print("Patch working! keys:", latents['pi_logits'].keys())
# ──────────────────────────────────────────

optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# ─────────────────────────────────────────────
# REWARD TRACKER
# ─────────────────────────────────────────────

LOG_REWARD = 4
PLANK_REWARD = 10
CRAFT_TABLE_REWARD = 100
PICK_REWARD = 150
STONE_REWARD = 8
COAL_REWARD = 1000

class RewardTracker:
    def __init__(self):
        # We now track the MAXIMUM ever reached to prevent drop/pickup exploits
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

    def get_inventory_count(self, info, item_name):
        total = 0
        for slot in info['inventory'].values():
            if item_name in slot['type']:  # ← 'in' instead of '=='
                total += slot['quantity']
        return total

    def process_item_reward(self, name, current, max_ever, cap, reward_val, hoard_limit, penalty_val):
        delta_reward = 0.0
        
        # ONLY reward if the agent reached a NEW lifetime high for this item
        if current > max_ever:
            print(f"  *** {name.upper()} NEW RECORD: {max_ever} -> {current} ***")
            
            # 1. Reward progress up to the cap
            items_to_reward = max(0, min(current, cap) - max_ever)
            delta_reward += items_to_reward * reward_val
            
            # 2. Hoarding Penalty (Only if they exceed hoard limit for the first time)
            if current > hoard_limit:
                new_excess = current - max(max_ever, hoard_limit)
                if new_excess > 0:
                    delta_reward -= new_excess * penalty_val
                    print(f"      [Penalty] Hoarding {name}: -{new_excess * penalty_val:.2f}")
        
        return delta_reward

    def compute(self, obs, info):
        reward = 0.0
        self.step_count += 1

        # slight penalty for being in inventory to encourage crafting but not staing in inv
        # if info.get('is_gui_open', False):
        #     # Only reward the first 100 steps of being in the menu
        #     if self.menu_timer < 50:
        #         reward += 0.005
        #         self.menu_timer += 1
        #     else:
        #         # After that, it becomes a penalty to prevent camping
        #         reward -= 0.01
        # else:
        #     self.menu_timer = 0
        # if not self.has_opened_inventory and info.get('is_gui_open', False):
        #     if self.max_logs > 0: # Only reward if they actually have something to craft!
        #         reward += 15.0
        #         self.has_opened_inventory = True
        #         print(" *** ACHIEVEMENT: Ready to Craft! +15.0 ***")

        # ── MOVEMENT CHECK EVERY 100 TURNS ──
        # if self.step_count % 1000 == 0:
        #     curr_x = info['player_pos']['x']
        #     curr_z = info['player_pos']['z']
            
        #     # 1. Calculate distance moved
        #     distance = abs(curr_x - self.prev_x) + abs(curr_z - self.prev_z)
            
        #     # 2. Calculate current progress sum
        #     # This sums all lifetime maximums to see if the agent "achieved" anything new
        #     current_total_progress = (self.max_logs + self.max_planks + self.max_sticks + 
        #                               self.max_stone + self.max_coal)
            
        #     # If crafting table or pickaxe were obtained, we add those to progress too
        #     if self.rewarded_crafting_table: current_total_progress += 1
        #     if self.rewarded_pickaxe: current_total_progress += 1

        #     # 3. Apply Penalty ONLY if movement is low AND no progress was made
        #     if distance < self.movement_threshold:
        #         if current_total_progress <= self.prev_total_progress:
        #             reward -= 10.0
        #             print(f"  [Penalty] TRUE IDLE: No movement and no mining/crafting: -10.0")
        #         else:
        #             print(f"  [Notice] Low movement but progress detected. Skipping penalty.")
            
        #     # Update trackers for the next 100-step check
        #     self.prev_x = curr_x
        #     self.prev_z = curr_z
        #     self.prev_total_progress = current_total_progress

        
        
        
        # Check if the agent is 'attacking' while looking at a log
        if info.get('is_attacking', False) and 'log' in info.get('looking_at_block', ''):
            reward += 0.05  # Tiny 'breadcrumb' reward to encourage hitting the tree
        # ── LOGS ──
        current_logs = self.get_inventory_count(info, 'log')
        reward += self.process_item_reward('log', current_logs, self.max_logs, 10, LOG_REWARD, 12, 2)
        self.max_logs = max(self.max_logs, current_logs)

        # ── PLANKS ──
        current_planks = self.get_inventory_count(info, 'planks')
        reward += self.process_item_reward('planks', current_planks, self.max_planks, 40, PLANK_REWARD, 48, 2)
        self.max_planks = max(self.max_planks, current_planks)

        # ── STICKS ── (The Bridge to the Pickaxe)
        current_sticks = self.get_inventory_count(info, 'stick')
        # Reward 2.0 for first 4 sticks, then only 0.1 for the next 8, then penalty
        reward += self.process_item_reward('stick', current_sticks, self.max_sticks, 4, 2.0, 12, 0.1)
        self.max_sticks = max(self.max_sticks, current_sticks)

        # ── STONE ──
        current_stone = self.get_inventory_count(info, 'cobblestone')
        reward += self.process_item_reward('stone', current_stone, self.max_stone, 32, STONE_REWARD, 64, 0.1)
        self.max_stone = max(self.max_stone, current_stone)

        # ── COAL ──
        current_coal = self.get_inventory_count(info, 'coal')
        reward += self.process_item_reward('coal', current_coal, self.max_coal, 8, COAL_REWARD, 16, 0.5)
        self.max_coal = max(self.max_coal, current_coal)

        # ── ONE-TIME MILESTONES ──
        if not self.rewarded_crafting_table and self.get_inventory_count(info, 'crafting_table') > 0:
            reward += CRAFT_TABLE_REWARD
            self.rewarded_crafting_table = True
            print(f"  *** CRAFTING TABLE OBTAINED ***")

        if not self.rewarded_pickaxe and self.get_inventory_count(info, 'wooden_pickaxe') > 0:
            reward += PICK_REWARD
            self.rewarded_pickaxe = True
            print(f"  *** PICKAXE OBTAINED ***")
        
        # ── RESOURCE WASTE PENALTY ──
        # Calculate how many items the agent has "lost" from its lifetime peak
        # This includes placing logs, throwing items, or losing them on death.
        # total_current_items = (current_logs + current_planks + current_sticks + 
        #                     current_stone + current_coal)

        # total_max_items = (self.max_logs + self.max_planks + self.max_sticks + 
        #                 self.max_stone + self.max_coal)
        # self.total_S_current_items = self.total_S_current_items

        # if total_current_items < total_max_items:
        #     lost_count = total_max_items - total_current_items
        #     # Penalty for each missing item. 
        #     # This makes 'placing' a log feel like a small 'hurt' signal.
        #     reward -= (lost_count * 0.5) 
        #     if self.step_count % 50 == 0: # Avoid log spam
        #         print(f"  [Penalty] Missing {lost_count} items from peak: -{lost_count * 0.5}")

        # ── SURVIVAL ──
        reward -= 0.01 
        if info.get('health', 20) <= 0:
            reward -= 10.0

        return reward

# ─────────────────────────────────────────────
# RETURNS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# ACTION + LOGPROB
# ─────────────────────────────────────────────
def get_action_and_logprob(policy, obs, memory):
    input_batched = dict_map(policy._batchify, obs)
    
    # Batchify memory the same way get_action does
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

    # Unbatchify and detach state_out
    state_out = recursive_tensor_op(lambda x: x[0], state_out)
    state_out = recursive_tensor_op(lambda x: x.detach(), state_out)

    return action, state_out, log_prob

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────
HOME_X, HOME_Y, HOME_Z = 4674.5, 63.0, -1573.5
IX, IY, IZ = int(HOME_X), int(HOME_Y), int(HOME_Z)

FIXED_SPAWN = [{
    "seed": 92641161,
    "position": [4674.5, -1573.5, 63.0] # [x, z, y] per your source code
}]

env = MinecraftSim(
    obs_size=(128, 128),
    seed=SET_SEED,
    # preferred_spawn_biome=NONE, 
    callbacks=[
        # This is the key to fixed spawning
        HardResetCallback(spawn_positions=FIXED_SPAWN),
        CommandsCallback(commands=[
            f'/setworldspawn {IX} {IY} {IZ}',
            f'/spawnpoint @a {IX} {IY} {IZ}',
            f'/tp @a {HOME_X} {HOME_Y} {HOME_Z}',
            '/clear'
        ]),
        RecordCallback(record_path="./output", fps=30, frame_type="pov")
    ]
)

# ─────────────────────────────────────────────
# TRAIN LOOP
# ─────────────────────────────────────────────
import random
print(f"Training on {DEVICE}")

for episode in range(NUM_EPISODES):
    # current_seed = random.randint(1, 1000000)
    
    # ── AUTO RESTART ON CRASH ──
    try:
        obs, info = env.reset()

        # for _ in range(2):
        #     obs, _, _, _, info = env.step(env.noop_action())

        # env.env.execute_cmd(f'/tp @p {HOME_X} {HOME_Y} {HOME_Z}')
        # action = env.noop_action()
        # obs, reward, terminated, truncated, info = env.step(action)
        # print(f"✅ Episode {episode+1} starting at fixed point: {HOME_X}, {HOME_Z}")
        # EXTRACT SEED HERE
        
        # print(f"Available info keys: {list(info.keys())}")
    except Exception as e:
        print(f"Environment crashed: {e}. Restarting...")
        try:
            env.close()
        except:
            pass
        

        env = MinecraftSim(
            obs_size=(128, 128),
            seed=SET_SEED,
            # preferred_spawn_biome="forest", 
            callbacks=[
                # This is the key to fixed spawning
                HardResetCallback(spawn_positions=FIXED_SPAWN),
                CommandsCallback(commands=[
                    f'/setworldspawn {IX} {IY} {IZ}',
                    f'/spawnpoint @a {IX} {IY} {IZ}',
                    f'/tp @a {HOME_X} {HOME_Y} {HOME_Z}',
                    '/clear'
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
            action, memory, log_prob = get_action_and_logprob(policy, obs, memory)
            obs, _, terminated, truncated, info = env.step(action)
            # if step == 0:
            #     print("Info keys:", info.keys())
            #     print("Info sample:", info)
        except Exception as e:
            print(f"Step failed: {e}. Ending episode early.")
            break

        reward = reward_tracker.compute(obs, info)
        total_reward += reward

        # ADD THIS BLOCK:
        if step % 500 == 0:
            print(f"    > Step {step:4d} | Current Reward: {total_reward:6.2f} | Inventory: Logs({reward_tracker.max_logs}) Planks({reward_tracker.max_planks})")
        
        rewards.append(reward)
        log_probs.append(log_prob)

        if step % MEMORY_RESET_EVERY == 0:
            memory = None

        if len(rewards) >= UPDATE_EVERY:
            returns = compute_returns(rewards)
            stacked = torch.stack(log_probs)
            loss    = -(stacked * returns).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            last_loss = loss.item()
            rewards   = []
            log_probs = []
            torch.cuda.empty_cache()

        if terminated or truncated:
            break

    print(
        f"Episode {episode+1}/{NUM_EPISODES} | "
        f"Steps {step+1} | "
        f"Reward {total_reward:.2f} | "
        f"Loss {last_loss:.4f}"
    )

    # ── SAVE BEST ──
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
            
        print(f"  ★ New best! Reward {best_reward:.2f} saved to policy_best.pt")

    if (episode + 1) % SAVE_EVERY == 0:
        path = f"./checkpoints/policy_ep{episode+1}.pt"
        torch.save(policy.state_dict(), path)
        print(f"Saved: {path}")

env.close()
print("Training complete.")