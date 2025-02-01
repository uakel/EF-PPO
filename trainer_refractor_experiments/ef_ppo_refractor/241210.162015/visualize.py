from plotwist import *
from matplotlib import pyplot as plt

data = make_nested_dict_from_sspe("log.sspe")
plt.title("Episode Length")
plt.plot(data["train/episode_length/mean"])
add_fig()
plt.title("Rewards")
plt.plot(data["train/episode_return/mean"])
add_fig()
plt.plot(data["train/constraint_return/mean"])
add_fig()
make("SagittalRefractor")
