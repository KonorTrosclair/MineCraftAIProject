from minestudio.data import RawDataset
from minestudio.data.minecraft.callbacks import ImageKernelCallback, ActionKernelCallback

dataset = RawDataset(
    dataset_dirs=['6xx'],  # Free Gameplay dataset
    modal_kernel_callbacks=[
        ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False),
        ActionKernelCallback(enable_prev_action=True, win_bias=1, read_bias=-1),
    ],
    win_len=128,
    split_ratio=0.9,
    shuffle_episodes=True,
)

print("Dataset loaded, first item keys:", dataset[0].keys())