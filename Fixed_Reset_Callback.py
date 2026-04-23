from minestudio.simulator.callbacks import FastResetCallback
import random

class FixedResetCallback(FastResetCallback):
    def __init__(self, fixed_x, fixed_y, fixed_z, **kwargs):
        # We still need biomes and range to satisfy the base class init
        super().__init__(**kwargs)
        self.fixed_x = fixed_x
        self.fixed_y = fixed_y
        self.fixed_z = fixed_z

    def before_reset(self, sim, reset_flag):
        if not sim.already_reset:
            return reset_flag
        
        # We replace the random biome search with a direct coordinate TP
        # Using standard /tp @a instead of /teleportbiome
        fast_reset_commands = [
            "/kill", 
            f"/time set {self.start_time}",
            f"/weather {self.start_weather}",
            "/kill @e[type=!player]",
            "/kill @e[type=item]",
            f"/tp @a {self.fixed_x} {self.fixed_y} {self.fixed_z}" # Forced Spot
        ]
        
        for command in fast_reset_commands:
            # Note: The source uses sim.env.execute_cmd
            sim.env.execute_cmd(command)
            
        return False # Tells the simulator NOT to do a hard reload