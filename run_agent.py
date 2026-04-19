from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from minestudio.models import VPTPolicy

policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_1x").to("cuda")
policy.eval()

env = MinecraftSim(
    obs_size=(128, 128),
    callbacks=[RecordCallback(record_path="./output", fps=30, frame_type="pov")]
)

memory = None
obs, info = env.reset()
for i in range(1200):
    action, memory = policy.get_action(obs, memory, input_shape='*')
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
print("Done! Check the ./output folder for the recorded video.")