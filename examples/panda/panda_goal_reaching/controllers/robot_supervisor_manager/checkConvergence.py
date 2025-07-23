import matplotlib.pyplot as plt
import numpy as np
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_data(text_path="./exports/", dest_path="./exports/"):
    file=open(text_path+"Episode-score.txt", "r")
    lines = file.read().splitlines()
    scores = list(map(float, lines))
    episode = list(range(1, 1+len(scores)))
    plt.figure()
    plt.title("Episode scores over episode")
    plt.plot(episode, scores, label='Raw data')
    simple_moving_average = moving_average(scores, 500)
    plt.plot(simple_moving_average, label='SMA500')
    plt.xlabel("episode")
    plt.ylabel("episode score")
    
    plt.legend()
    plt.savefig(dest_path+'trend.png')
    print("Last SMA500:",np.mean(scores[-500:]))

if __name__ == '__main__':
    plot_data()
