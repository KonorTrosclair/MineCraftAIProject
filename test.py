import minestudio
from minestudio.simulator import MinecraftSim

# Create a dummy sim
sim = MinecraftSim(action_type="env")

# 1. See EVERY method available on the sim object
print("--- ALL METHODS ---")
print(dir(sim))

# 2. See methods of the underlying environment (where most commands live)
print("\n--- UNDERLYING ENV METHODS ---")
if hasattr(sim, 'env'):
    print(dir(sim.env))

# 3. See the 'unwrapped' core (the rawest form of the simulator)
print("\n--- UNWRAPPED CORE METHODS ---")
print(dir(sim.unwrapped))