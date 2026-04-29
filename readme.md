Training an AI agent to play Minecraft though Reinforcement Learning

Contributers:
    Konor Trosclair, Kasen Sinclair

Geting Started:

    Clone the Repository:
        git clone https://github.com/KonorTrosclair/MineCraftAIProject
    You will need Python version 3.10
    Install wsl Ubuntu (if on windows system):
        wsl --install (will require restart of PC)

    Launch the wsl instance by typing (wsl) in terminal.

    while in wsl enviorment do the following:
    Install Conda:
        Get shell script:
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        Run shell script:
            bash Miniconda3-latest-Linux-x86_64.sh

    create a Conda environment:
        conda create -n <name> python-3.10
        
    launch the Conda environment:
        cond activate <name>  (to exit: conda deactivate)
    
    Dependencies:
        Dependencies can be installed using Pip install command (pip install <dependency>):
            - torch
            - minestudio
    To Run:
        - To run the training script type (python3 train_agent.py)
        - To run the run script type (python3 run_agent.py)
        - to check for the reward of the best run type (python3 get_reward.py)


Special notes:
    
    Config:
        At the top of the tain_agent.py script there are config variables (line 21 down) Here you can change variables like learning rate episodes steps etc...
        UPDATE_EVERY:
            - This variable is very important and primarily dependent on your devices VRAM. For reference the device that we used to train the agent on had a NVIDIA RTX 3080 with 10 GB of VRAM.
            - The value 200 for the UPDATE_EVERY worked well with 10gb of VRAM and is encouraged to increase or decrease if more or less VRAM is available.
            - A higher value here is more favorable so the agent has a longer short term memory.
    Reward Tracker:
        - There are constants to determine reward amounts for obtaining different objects (line 63) 
        - There is also a function compute that is responsible for calculating rewards this can be changed to direct the agent in the preferred direction. (currently it is set up to get a crafting table)
        
    File saving:
        The script saves files in the form of logs, videos, and policy checkpoints.
            LOGS:
                - logs are saved in the logs folder and contain info regarding the Minecraft launch instance this is used for debugging the Minecraft environment.
            VIDEOS:
                - videos are saved as .MP4 files and are saved int he output folder.
                - Videos show what the agent done in the recent episode 
                - Videos also only save up to 8 or 9 before overwriting previous videos this is due to the Minecart environment crashing after every 8 episodes.
            CHECKPOINTS:
                - checkpoints are saved to the folder checkpoints and are saved as .pt files and contain the weights of saved runs.
                - checkpoints are saved every 5 episodes and every new record reward (example: previouse recod was 100 new record was 101 saved the episodes policy with reward 101
    Other Notes:
        CRASHING:
            - The Minecraft environment crashes after every 8 to 9 episodes this is to be expected as the Minestudio Minecraft process is known to crash after extended run times
            - To combat this the environment is set to restart upon crashing.
        get_reward.py
            - The get_reward.py scripty is designated to read from the best checkpoint to see what the reward of that checkpoint is before loading it
