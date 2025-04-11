import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_metrics(rewards, epsilons, save_prefix='ddqn'):
    episodes = np.arange(len(rewards))

    plt.figure(figsize=(12, 5))

    # Reward Plot
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, label='Episode Reward')
    avg_rewards = [np.mean(rewards[max(0, i - 100):i + 1]) for i in range(len(rewards))]
    plt.plot(episodes, avg_rewards, label='Moving Avg (100)', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.legend()

    # Epsilon Plot
    plt.subplot(1, 2, 2)
    plt.plot(episodes, epsilons)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_metrics.png")
    plt.show()

if __name__ == "__main__":
    with open("logs.pkl", "rb") as f:
        rewards, epsilons = pickle.load(f)
    plot_metrics(rewards, epsilons)
