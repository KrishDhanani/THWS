import torch


# TODO 1: complete the code below
def linreg_closed(data):
    """Closed form solution for parameters of linear model.
    y = theta_0 + theta_1 x
    
    Args:
        data: (n, 2) torch tensor; inputs - 1st col, outputs 2nd col
    """

    # Closed Form
    # ∂L/∂θ = 0
    # ↓ solve algebraically
    # θ = (XᵀX)⁻¹Xᵀy

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
  # Gradient Descent
  # compute ∂L/∂θ
  # ↓ take small steps
  # θ ← θ − lr · ∂L/∂θ

  
  # initialize theta
  theta_0 = torch.tensor(0.0)
  theta_1 = torch.tensor(0.0)
  
  # monitor losses
  losses = torch.empty((n_epochs, 1))
  
  # initial prediction and loss
  X = data[:, 0]
  y = data[:, -1]
  y_pred = theta_0 + theta_1 * X
  losses[0] = mse(y, y_pred)
  
  # grad descent
  for epoch in list(range(n_epochs)):

    n = X.shape[0]
    theta_0 = theta_0 - lr * (-2/n) * torch.sum(y-y_pred)
    theta_1 = theta_1 - lr * (-2/n) * torch.sum((y-y_pred) * X)

    # monitor losses
    y_pred = theta_0 + theta_1 * X
    losses[epoch] = mse(y, y_pred)
  
  # final parameters
  theta_0 = theta_0.item()
  theta_1 = theta_1.item()

  # Return output
  return theta_0, theta_1, losses



# TODO 4: complete the code below
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
  theta_0 = torch.tensor(0.0)
  theta_1 = torch.tensor(0.0)
  
  # monitor losses
  n = data.size(0)    # same as data.shape[0]
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  x = data[:, 0]
  y = data[:, -1]
  y_pred = theta_0 + theta_1*x
  losses[0] = mse(y, y_pred)
  
  # grad descent
  idx = torch.randperm(n)
  lidx = 0
  for epoch in list(range(n_epochs)):
    for batch in range(n_batches):

      start = batch * bs
      end = (batch + 1) * bs
      batch_idx = idx[start:end]

      x_batch = x[batch_idx]
      y_batch = y[batch_idx]
      n_batch = len(x_batch)

      n = len(x_batch)
      y_pred = theta_0 + theta_1 * x_batch
      theta_0 = theta_0 - lr * (-2 / n) * torch.sum(y_batch - y_pred) 
      theta_1 = theta_1 - lr * (-2 / n) * torch.sum((y_batch - y_pred)*x_batch)

      # monitor losses
      y_pred_full = theta_0 + theta_1 * x
      losses[lidx] = mse(y, y_pred_full)
      lidx += 1
  
  # final parameters
  theta_0 = theta_0
  theta_1 = theta_1 

  # Return output
  return theta_0, theta_1, losses
