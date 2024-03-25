from matplotlib import pyplot as plt
from IPython import display
import torch


def plot_rewards(episode_rewards, epsilon_values, is_ipython, show_results=False):
    plt.figure(1)
    if show_results:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward (duration)")
    plt.plot(episode_rewards)
    if len(episode_rewards) >= 100:
        rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float)
        means = rewards_tensor.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if not show_results:
        plt.figure(2)
        plt.title("Epsilon")
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon value")
        plt.plot(epsilon_values)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_results:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
