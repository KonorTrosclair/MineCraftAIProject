import torch

# Define the path to your best checkpoint
checkpoint_path = './checkpoints/policy_best.pt'

try:
    # Load the file (map_location='cpu' ensures it works even without a GPU)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        # Access the best_reward key
        if 'best_reward' in checkpoint:
            val = checkpoint['best_reward']
            print(f"\n" + "="*30)
            print(f"  🏆 BEST REWARD: {val:.2f}")
            print("="*30 + "\n")
        else:
            print("Successfully loaded, but 'best_reward' key was not found.")
            print("Available keys:", list(checkpoint.keys()))
    else:
        print("This file contains a raw state_dict, not a metadata dictionary.")

except FileNotFoundError:
    print(f"Error: The file '{checkpoint_path}' does not exist yet.")
except Exception as e:
    print(f"An error occurred: {e}")