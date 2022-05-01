import matplotlib.pyplot as plt
import numpy as np

record = open("result3.txt", 'r')
reward = []
x = []
for line in record.readlines():
    l = line.split(' ')
    l[-1] = l[-1].replace('\n', '')
    reward.append(float(l[-1]))
    l[1] = l[1].replace(',', '')
    x.append(float(l[1]))

plt.plot(x, reward)
plt.xlabel("Time Steps")
plt.ylabel("Accumulated Rewards")
plt.title("3 Million Timesteps Training Results")
plt.legend()
plt.savefig("result2.png")
