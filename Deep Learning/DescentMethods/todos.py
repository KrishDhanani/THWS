"""Gradient descent exploration - todos"""

import torch

# TODO 1: linear model
def linear_model(x, theta):
  """Predictions of linear model.
   
  Args:
    x: (n, 1) torch.tensor with inputs
    theta: (2, 1) torch.tensor with parameters

  Output:
    y: (n, 1) torch.tensor with linear predictions
  """

  # prepend x by a zeros to match the dims of theta
  x = 
  y =
  return y

def mse(y, y_predict):
  """Mean squared error loss.
   
  Args:
    y, y_predict: (n, 1) torch.tensors with targets and predictions

  Output:
    mse: (scalar)
  """
  mse =
  return mse

# TODO 2: closed form solution for linear regression
def linreg_closed(x, y):
  """Closed form solution for parameters of linear model.
  y = theta.T x
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (n, 1) torch.tensor with linear predictions

  Output:
    theta: (2, 1) torch.tensor with parameters
  """
  
  x0 = torch.ones_like(x)
  x = torch.cat((x0,x), dim=1)
  
  th = torch.pinverse(x.T @ x) 
  th = th @ x.T @ y
  
  # Return output
  return th


# TODO 3: complete the code below
def lin_grads(x, y, theta):
  """Gradient for parameters of linear model.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
  """
  # prepend x by a zeros to match the dims of theta
  x = 
  grad = 
  return grad


def grad_descent(x,y, theta, lr, grad_func, model, n_epochs=1):
  """Gradient descent for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
    lr: (scalar) learning rate
    grad_func: function to calculate gradients
    model: model for predictions
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # monitor thetas
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  losses = torch.empty((n_epochs+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)
  
  # grad descent
  for ....

    # monitor thetas and losses
    thetas[:,epoch+1] = 
    losses[epoch+1] = 
  
  # Return output
  return thetas, losses


# TODO 4: complete the code below
def stochastic_grad_descent(x,y, theta, lr, bs, grad_func, model, n_epochs=1):
  """Stochastic gradient descent for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
    lr: (scalar) learning rate
    bs: batch size
    grad_func: function to calculate gradients
    model: model for predictions
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # monitor thetas
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  for ...

      # monitor losses
      losses[...] = 
 
    # monitor thetas
    thetas[...] = 

  # Return output
  return thetas, losses


# TODO 5: complete the code below
def sgd_momentum(x,y, theta, lr, beta, bs, grad_func, model, n_epochs=1):
  """SGD with momentum for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
    lr: (scalar) learning rate
    beta: momentum smoothing parameter
    bs: batch size
    grad_func: function to calculate gradients
    model: model for predictions
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # monitor thetas
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  for ...

      # monitor losses
      losses[...] = 
 
    # monitor thetas
    thetas[...] = 

  # Return output
  return thetas, losses



# TODO 6: complete the code below
def sgd_nestorov(x,y, theta, lr, beta, bs, grad_func, model, n_epochs=1):
  """SGD with accelerated momentum for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
    lr: (scalar) learning rate
    beta: momentum smoothing parameter
    bs: batch size
    grad_func: function to calculate gradients
    model: model for predictions
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # monitor thetas
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  for ...

      # monitor losses
      losses[...] = 
 
    # monitor thetas
    thetas[...] = 

  # Return output
  return thetas, losses


# TODO 7: complete the code below
def ADAM(x,y, theta, lr, beta, gamma, bs, grad_func, model, n_epochs=1):
  """ADAM for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
    lr: (scalar) learning rate
    beta: momentum smoothing parameter
    gamma: momentum for variance
    bs: batch size
    grad_func: function to calculate gradients
    model: model for predictions
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # monitor thetas
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  for ...

      # monitor losses
      losses[...] = 
 
    # monitor thetas
    thetas[...] = 

  # Return output
  return thetas, losses
