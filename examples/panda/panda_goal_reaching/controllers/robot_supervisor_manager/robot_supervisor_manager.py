"""
More runners for RL algorithms can be added here.
"""
import os
import DDPG_runner

# Modify these constants if needed.
STEPS_PER_EPISODE = 300 
MOTOR_VELOCITY = 10 
EPISODE_LIMIT = 50000
SAVE_MODELS_PERIOD = 200

def create_path(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

if __name__ == '__main__':
    print("Hello")
    # dest of models' checkpoints
    create_path("./tmp/ddpg/")

    # dest of traing trend (text file & trend plot)
    create_path("./exports/")

    # pass a path to load the pretrained models, and pass "" for training from scratch
    load_path = "./tmp/ddpg/" 
    DDPG_runner.run(load_path)