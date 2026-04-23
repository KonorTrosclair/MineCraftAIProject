from omegaconf import OmegaConf
from minestudio.online.rollout.start_manager import start_rolloutmanager
from minestudio.online.trainer.start_trainer import start_trainer
from minestudio.models import VPTPolicy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    RewardsCallback, FastResetCallback,
    JudgeResetCallback, SummonMobsCallback,
    CommandsCallback, MaskActionsCallback,
)

def policy_generator():
    return VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_from_early_game_2x")

def env_generator():
    return MinecraftSim(
        obs_size=(128, 128),
        preferred_spawn_biome="plains",
        action_type="agent",
        timestep_limit=300,
        callbacks=[
            SummonMobsCallback([{"name": "cow", "number": 10, "range_x": [-5, 5], "range_z": [-5, 5]}]),
            MaskActionsCallback(inventory=0),
            RewardsCallback([{"event": "kill_entity", "objects": ["cow"], "reward": 1.0, "identity": "kill_cow", "max_reward_times": 10}]),
            CommandsCallback(commands=["/give @p minecraft:iron_sword 1"]),
            FastResetCallback(biomes=["plains"], random_tp_range=1000),
            JudgeResetCallback(300),
        ],
    )

online_cfg = OmegaConf.create({
    "trainer_name": "PPOTrainer",
    "detach_rollout_manager": True,
    "rollout_config": {
        "num_rollout_workers": 1,
        "num_gpus_per_worker": 1.0,
        "num_cpus_per_worker": 1,
        "fragment_length": 256,
        "to_send_queue_size": 6,
        "worker_config": {
            "num_envs": 4,
            "batch_size": 4,
            "restart_interval": 3600,
        },
        "replay_buffer_config": {
            "max_chunks": 1200,
            "max_reuse": 2,
            "max_staleness": 2,
            "fragments_per_report": 10,
            "fragments_per_chunk": 1,
            "database_config": {
                "path": "output/replay_buffer_cache",
                "num_shards": 4,
            },
        },
        "episode_statistics_config": {},
    },
    "train_config": {
        "num_workers": 1,
        "num_gpus_per_worker": 1.0,
        "num_iterations": 4000,
        "learning_rate": 0.00002,
        "weight_decay": 0.04,
        "batch_size_per_gpu": 1,
        "batches_per_iteration": 16,
        "epochs_per_iteration": 1,
        "clip_range": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "gamma": 0.9999,
        "checkpoint_interval": 50,
        "checkpoint_path": "output/checkpoints",
    },
})

start_rolloutmanager(policy_generator, env_generator, online_cfg)
start_trainer(policy_generator, env_generator, online_cfg)