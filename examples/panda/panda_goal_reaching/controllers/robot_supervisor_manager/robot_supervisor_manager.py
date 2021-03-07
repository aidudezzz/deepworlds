"""
More runners for RL algorithms can be added here.
"""
import DDPG_runner
import os

def create_path(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

# dest of models' checkpoints
create_path("./tmp/ddpg/")

# dest of traing trend (text file & trend plot)
create_path("./exports/")

# pass a path to load the pretrained models, and pass "" for training from scratch
load_path = "./tmp/ddpg/" 
DDPG_runner.run(load_path)