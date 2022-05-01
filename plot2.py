import numpy as np
import matplotlib.pyplot as plt

reward1 = np.load("acc_reward1.npy")
reward2 = np.load("acc_reward2.npy")
reward3 = np.load("acc_reward3.npy")
reward_normal1 = np.load("acc_reward_normal1.npy")
reward_normal2 = np.load("acc_reward_normal2.npy")
reward_normal3 = np.load("acc_reward_normal3.npy")

actor_loss1 = np.load("actor_loss1.npy")
actor_loss2 = np.load("actor_loss2.npy")
actor_loss3 = np.load("actor_loss3.npy")
actor_loss_normal1 = np.load("actor_loss_normal1.npy")
actor_loss_normal2 = np.load("actor_loss_normal2.npy")
actor_loss_normal3 = np.load("actor_loss_normal3.npy")

critic_loss1 = np.load("critic_loss1.npy")
critic_loss2 = np.load("critic_loss2.npy")
critic_loss3 = np.load("critic_loss3.npy")
critic_loss_normal1 = np.load("critic_loss_normal1.npy")
critic_loss_normal2 = np.load("critic_loss_normal2.npy")
critic_loss_normal3 = np.load("critic_loss_normal3.npy")

actor_loss1 = actor_loss1[actor_loss1.nonzero()]
actor_loss2 = actor_loss2[actor_loss2.nonzero()]
actor_loss3 = actor_loss3[actor_loss3.nonzero()]
actor_loss_normal1 = actor_loss_normal1[actor_loss_normal1.nonzero()]
actor_loss_normal2 = actor_loss_normal2[actor_loss_normal2.nonzero()]
actor_loss_normal3 = actor_loss_normal3[actor_loss_normal1.nonzero()]

critic_loss1 = critic_loss1[critic_loss1.nonzero()]
critic_loss2 = critic_loss2[critic_loss2.nonzero()]
critic_loss3 = critic_loss3[critic_loss3.nonzero()]
critic_loss_normal1 = critic_loss_normal1[critic_loss_normal1.nonzero()]
critic_loss_normal2 = critic_loss_normal2[critic_loss_normal2.nonzero()]
critic_loss_normal3 = critic_loss_normal3[critic_loss_normal3.nonzero()]

mean_reward = np.mean(np.array([reward1, reward2, reward3]), axis=0)
std_reward = np.std(np.array([reward1, reward2, reward3]), axis=0)
mean_normal_reward = np.mean(np.array([reward_normal1, reward_normal2, reward_normal3]), axis=0)
std_normal_reward = np.std(np.array([reward_normal1, reward_normal2, reward_normal3]), axis=0)

mean_actor_loss = -1 * np.mean(np.array([actor_loss1, actor_loss2, actor_loss3]), axis=0)
std_actor_loss = np.std(np.array([actor_loss1, actor_loss2, actor_loss3]), axis=0)
mean_normal_actor_loss = -1 * np.mean(np.array([actor_loss_normal1, actor_loss_normal2, actor_loss_normal3]), axis=0)
std_normal_actor_loss = np.std(np.array([actor_loss_normal1, actor_loss_normal2, actor_loss_normal3]), axis=0)

mean_critic_loss = np.mean(np.array([critic_loss1, critic_loss2, critic_loss3]), axis=0)
std_critic_loss = np.std(np.array([critic_loss1, critic_loss2, critic_loss3]), axis=0)
mean_normal_critic_loss = np.mean(np.array([critic_loss_normal1, critic_loss_normal2, critic_loss_normal3]), axis=0)
std_normal_critic_loss = np.std(np.array([critic_loss_normal1, critic_loss_normal2, critic_loss_normal3]), axis=0)

_, _, bars1 = plt.errorbar(x=np.arange(mean_reward.shape[0]) * 1000, y=mean_reward, yerr=std_reward, label="With Reward Scaling")
_, _, bars2 = plt.errorbar(x=np.arange(mean_normal_reward.shape[0]) * 1000, y=mean_normal_reward, yerr=std_normal_reward, label="Without Reward Scaling")
[bar.set_alpha(0.1) for bar in bars1]
[bar.set_alpha(0.1) for bar in bars2]
plt.xlabel("Time step")
plt.ylabel("Accumulated Reward")
plt.title("Accumulated Reward with Reward Scaling")
plt.legend()
plt.savefig("reward.png")
plt.show()

_, _, bars1 = plt.errorbar(x=np.arange(mean_actor_loss.shape[0]) * 4001, y=mean_actor_loss, yerr=std_actor_loss, label="With Reward Scaling")
_, _, bars2 = plt.errorbar(x=np.arange(mean_normal_actor_loss.shape[0]) * 4001, y=mean_normal_actor_loss, yerr=std_normal_actor_loss, label="Without Reward Scaling")
[bar.set_alpha(0.3) for bar in bars1]
[bar.set_alpha(0.3) for bar in bars2]
plt.xlabel("Time step")
plt.ylabel("Actor Loss")
plt.title("Actor Loss with Reward Scaling")
plt.legend()
plt.savefig("actor_loss.png")
plt.show()

_, _, bars1 = plt.errorbar(x=np.arange(mean_critic_loss.shape[0]) * 4001, y=mean_critic_loss, yerr=std_critic_loss, label="With Reward Scaling")
_, _, bars2 = plt.errorbar(x=np.arange(mean_normal_critic_loss.shape[0]) * 4001, y=mean_normal_critic_loss, yerr=std_normal_critic_loss, label="Without Reward Scaling")
[bar.set_alpha(0.3) for bar in bars1]
[bar.set_alpha(0.3) for bar in bars2]
plt.xlabel("Time step")
plt.ylabel("Critic Loss")
plt.title("Critic Loss with Reward Scaling")
plt.legend()
plt.savefig("critic_loss.png")
plt.show()
