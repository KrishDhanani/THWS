"""Helper functions for shallow net exercise"""


import numpy as np
import matplotlib.pyplot as plt
import torch


# Plot data
def plot_data(x, y, y_pred=torch.empty(0,0), title="Data plot"): 
  fig, ax = plt.subplots(figsize=(5,5))
  ax.scatter(x,y)
  if torch.numel(y_pred) != 0:
    ax.scatter(x, y_pred, c='red')
  ax.set_xlabel('Input'); ax.set_ylabel('Output')
  ax.set_title(title)
  plt.show()


# Plot losses
def plot_loss(losses, title="Loss plot"): 
  fig, ax = plt.subplots(figsize=(5,5))
  ax.plot(losses)
  ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
  ax.set_title(title)
  plt.show()
