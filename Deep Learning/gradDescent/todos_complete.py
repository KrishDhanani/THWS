import torch


# TODO 1: complete the code below
def linreg_closed(data):
    """Closed form solution for parameters of linear model.
    y = theta_0 + theta_1 x
    
    Args:
        data: (n, 2) torch tensor; inputs - 1st col, outputs 2nd col
    """
    x = data[:, 0]
    Y = data[:, -1]

    ones = torch.ones(x.shape[0])
    X = torch.stack([ones, x], dim=1)  # (n, 2)

    XtX = X.T @ X                      # (2, 2)
    XtY = X.T @ Y                      # (2,)
    theta = torch.linalg.solve(XtX, XtY)  # (A, B) A^-1 automatically internally

    theta_0 = theta[0]
    theta_1 = theta[1]

    return theta_0, theta_1


# TODO 2: complete the code below
def mse(y, y_predict):
  """Calculate the mean square error between targets and predictions.
   
  Args:
    y, y_predict: targets and predictions
  """

  mse = torch.mean((y_predict - y) ** 2)
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
  theta_0 = data[:, 0:1]
  theta_1 = data[:, 1:2]
  
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