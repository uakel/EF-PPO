from plotwist import *
from matplotlib import pyplot as plt

data = make_nested_dict_from_sspe("log.sspe")
plt.plot(data["train/episode_length/mean"])
plt.savefig("episode_length.png")



