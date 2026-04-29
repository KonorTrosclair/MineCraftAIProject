from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, HardResetCallback, CommandsCallback
from minestudio.models import VPTPolicy
import torch

DEVICE = "cuda"
HOME_X, HOME_Y, HOME_Z = 4674.5, 63.0, -1573.5
IX, IY, IZ = int(HOME_X), int(HOME_Y), int(HOME_Z)

FIXED_SPAWN = [{
    "seed": 92641161,
    "position": [4674.5, -1573.5, 63.0]
}]

policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_1x").to(DEVICE)


checkpoint = torch.load('./checkpoints/policy_best.pt')
if isinstance(checkpoint, dict) and 'policy' in checkpoint:
    policy.load_state_dict(checkpoint['policy'])
    print(f"Loaded best checkpoint, reward was: {checkpoint['best_reward']:.2f}")
else:
    policy.load_state_dict(checkpoint)

policy.eval()

env = MinecraftSim(
    obs_size=(128, 128),
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

memory = None
obs, info = env.reset()
for i in range(1200):
    action, memory = policy.get_action(obs, memory, input_shape='*')
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
print("Done! Check the ./output folder for the recorded video.")