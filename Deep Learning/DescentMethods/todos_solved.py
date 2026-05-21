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
  x_1 = torch.ones_like(x)
  x = torch.cat((x_1,x), dim=1)  # returns x with dim(n,2)
  y = x @ theta
  return y

def mse(y, y_predict):
  """Mean squared error loss.
   
  Args:
    y, y_predict: (n, 1) torch.tensors with targets and predictions

  Output:
    mse: (scalar)
  """
  mse = torch.mean((y_predict-y) ** 2)
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
  
  th = torch.pinverse(x.T @ x) # for getting inverse 
  th = th @ x.T @ y   # θ=((XTX)^−1)XTy
  
  # Return output
  return th



# TODO 3: complete the code below
def lin_grads(x, y, theta):
  """Gradient for parameters of linear model.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (n, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)

  J.J.
  HERE see mse and we take gradient of 'L' with respect to the "theta". and here eq. for y_pred = theta * x;  grad: ∂θ/∂L​ = (2/n​)X^T(Xθ−y)
  """
  # prepend x by a zeros to match the dims of theta
  x_1 = torch.ones_like(x)
  x = torch.cat([x_1,x], dim=1)
  n = len(x)
  grad = (2/n) * x.T @ (x @ theta - y) 
  return grad


def grad_descent(x,y, theta, lr, grad_func, model, n_epochs=1):
  """Gradient descent for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (n, 1) torch.tensor with outputs
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
  for epoch in range(n_epochs):

    # monitor thetas and losses
    theta = thetas[:, epoch:epoch+1]  # get current theta (2,1)
    grad = grad_func(x, y, theta)     # compute gradient
    thetas[:, epoch+1:epoch+2] = theta - lr * grad  # update theta
    
    y_pred = model(x, thetas[:, epoch+1:epoch+2])   # predict with new theta
    losses[epoch+1] = mse(y, y_pred)                # store loss
  
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

    for each epoch:
    shuffle data
    for each batch of size bs:
        compute gradient on batch only
        update theta with momentum
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
  for epoch in range(n_epochs):
      for batch in range(n_batches):

        start = batch * bs 
        end = (batch + 1) * bs
        batch_idx = x[start:end]

        theta = thetas[:, epoch:epoch+1] 
        grad = grad_func(x, y, theta)     
        thetas[:, epoch+1:epoch+2] = theta - lr * grad 

        y_pred = model(x, thetas[:, epoch+1:epoch+2])   
        losses[epoch+1] = mse(y, y_pred)  

  # Return output
  return thetas, losses