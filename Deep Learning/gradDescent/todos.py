"""3.2.1 Gradient descent - TODOs.

Complete the code here below.
"""

# import libraries - do not use numpy!
import torch


# TODO 1: complete the code below
def linreg_closed(data):
  """Closed form solution for parameters of linear model.
  y = theta_0 + theta_1 x
  
  Args:
    data: (n, 2) torch tensor; inputs - 1st col, outputs 2nd col
  """
  
  theta_0 = 
  theta_1 = 

  # Return output
  return theta_0, theta_1


# TODO 2: complete the code below
def mse(y, y_predict):
  """Calculate the mean square error between targets and predictions.
   
  Args:
    y, y_predict: targets and predictions
  """

  mse = 

  return mse


# TODO 3: complete the code below
def linreg_gd(data, lr, n_epochs=1000):
  """Gradient descent solution for parameters of linear model.
  y = theta_0 + theta_1 x
  
  Args:
    data: (n, 2) torch tensor; inputs - 1st col, outputs 2nd col
    lr: (scalar) learning rate
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
    # initialize theta
  theta_0 = 
  theta_1 = 
  
  # monitor losses
  losses = torch.empty((n_epochs, 1))
  
  # initial prediction and loss
  y_pred = 
  losses[0] = mse(y, y_pred)
  
  # grad descent
  for epoch in list(range(n_epochs)):

    theta_0 = 
    theta_1 = 

    # monitor losses
    y_pred = 
    losses[epoch] = mse(y, y_pred)
  
  # final parameters
  theta_0 = 
  theta_1 = 

  # Return output
  return theta_0, theta_1, losses


# TODO 3: complete the code below
def linreg_sgd(data, bs, lr, n_epochs=1000):
  """Stochastic gradient descent solution for parameters of linear model.
  y = theta_0 + theta_1 x
  
  Args:
    data: (n, 2) torch tensor; inputs - 1st col, outputs 2nd col
    bs: batch size
    lr: (scalar) learning rate
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # initialize theta
  theta_0 = 
  theta_1 = 
  
  # monitor losses
  n = data.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = theta_0 + theta_1*x
  losses[0] = mse(y, y_pred)
  
  # grad descent
  idx = torch.randperm(n)
  lidx = 0
  for epoch in list(range(n_epochs)):
    for batch in 

      theta_0 = 
      theta_1 = 

      # monitor losses
      y_pred = 
      losses[lidx] = mse(y, y_pred)
  
  # final parameters
  theta_0 = 
  theta_1 = 

  # Return output
  return theta_0, theta_1, losses

