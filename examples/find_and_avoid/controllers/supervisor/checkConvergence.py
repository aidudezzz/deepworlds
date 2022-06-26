import matplotlib.pyplot as plt
import numpy as np
import time
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def Plot(text_path="./exports/", dest_path="./exports/"):
    fp=open(text_path+"Episode-score.txt", "r")
    lines = fp.read().splitlines()
    scores = list(map(float, lines))
    episode = list(range(1, 1+len(scores)))
    plt.figure()
    plt.title("Episode scores over episode")
    plt.plot(episode, scores, label='Raw data')
    SMA = moving_average(scores, 50)
    plt.plot(SMA, label='SMA50')
    plt.xlabel("episode")
    plt.ylabel("episode score")
    
    plt.legend()
    plt.savefig(dest_path+'trend.png')
    plt.close()

if __name__ == '__main__':
    while(1):
        
        Plot()
        time.sleep(60)