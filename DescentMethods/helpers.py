"""Helper functions for descent explorations"""

import torch
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# simple regression data
data = torch.tensor([[0.03,0.19,0.34,0.46,0.78,0.81,1.08,1.18,1.39,1.60,1.65,1.90],
                 [0.67,0.85,1.05,1.00,1.40,1.50,1.30,1.54,1.55,1.68,1.73,1.60]]).T

gabor_data = torch.tensor([[-1.920e+00,-1.422e+01,1.490e+00,-1.940e+00,-2.389e+00,-5.090e+00,
                 -8.861e+00,3.578e+00,-6.010e+00,-6.995e+00,3.634e+00,8.743e-01,
                 -1.096e+01,4.073e-01,-9.467e+00,8.560e+00,1.062e+01,-1.729e-01,
                  1.040e+01,-1.261e+01,1.574e-01,-1.304e+01,-2.156e+00,-1.210e+01,
                 -1.119e+01,2.902e+00,-8.220e+00,-1.179e+01,-8.391e+00,-4.505e+00],
                  [-1.051e+00,-2.482e-02,8.896e-01,-4.943e-01,-9.371e-01,4.306e-01,
                  9.577e-03,-7.944e-02 ,1.624e-01,-2.682e-01,-3.129e-01,8.303e-01,
                  -2.365e-02,5.098e-01,-2.777e-01,3.367e-01,1.927e-01,-2.222e-01,
                  6.352e-02,6.888e-03,3.224e-02,1.091e-02,-5.706e-01,-5.258e-02,
                  -3.666e-02,1.709e-01,-4.805e-02,2.008e-01,-1.904e-01,5.952e-01]]).T


# plot data and model
def plot_model(x,y,theta,model,title=None):
  x_model = torch.arange(x.min()-0.5,x.max()+0.5,0.01)
  y_model = model(x_model[:,None],theta)

  fig, ax = plt.subplots()
  fig.set_size_inches(6,6)
  ax.plot(x,y,'bo')
  ax.plot(x_model,y_model,'m-')
  ax.set_xlim([x.min()-0.5,x.max()+0.5]);ax.set_ylim([y.min()-0.5,y.max()+0.5])
  ax.set_xlabel('x'); ax.set_ylabel('y')
  # ax.set_aspect('equal')
  if title is not None:
    ax.set_title(title)
  plt.show()


# plot loss countours
def loss_contour(x,y,model,loss_func,type=0,title=None,theta_list=None):
  t_init = theta_list[:,0]
  # t0 = torch.arange(t_init[1]-0.6,t_init[1]+0.6,0.1) # theta_1
  # t1 = torch.arange(t_init[0]-0.6,t_init[0]+1.1,0.1) # theta_0
  if type == 0:
    t0 = torch.linspace(-0.1,1.2,100) # theta_1
    t1 = torch.linspace(-0.1,1.2,100) # theta_0
  else:
    t0 = torch.linspace(2.4,22.5,100) # theta_1
    t1 = torch.linspace(-10.1,10.1,100) # theta_0

  grid_t0, grid_t1 = torch.meshgrid(t1, t0, indexing='ij')

  loss = torch.zeros_like(grid_t0)
  for i in range(grid_t0.shape[0]):
    for j in range(grid_t0.shape[1]):
      y_pred = model(x,torch.tensor([[grid_t0[i,j],grid_t1[i,j]]]).T)
      loss[i,j] = loss_func(y,y_pred)

  fig,ax = plt.subplots()
  fig.set_size_inches(6,6)
  ax.contourf(grid_t0.detach().numpy(),grid_t1.detach().numpy(),loss.detach().numpy(),256)
  ax.contour(grid_t0.detach().numpy(),grid_t1.detach().numpy(),loss.detach().numpy(),40,colors="gray")
  if theta_list is not None:
    ax.plot(theta_list[0,:], theta_list[1,:],'ro-')
  if title is not None:
    ax.set_title(title)
  # ax.set_ylim([t_init[1]-0.5,t_init[1]+0.5])
  # ax.set_xlim([t_init[0]-0.5,t_init[0]+1.])
  ax.set_ylim([t0[1],t0[99]])
  ax.set_xlim([t1[1],t1[99]])
  ax.set_xlabel(r'$\theta_0$'); ax.set_ylabel(r'$\theta_1$')
  plt.show()


# Plot losses
def plot_loss(losses, title="Loss plot"): 
  fig, ax = plt.subplots(figsize=(5,5))
  ax.plot(losses)
  ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
  ax.set_title(title)
  plt.show()

# Gabor model
def gabor_model(x,theta):
  sin_component = torch.sin(theta[0,0] + 0.06 * theta[1,0] * x)
  gauss_component = torch.exp(-(theta[0,0] + 0.06 * theta[1,0] * x) * (theta[0,0] + 0.06 * theta[1,0] * x) / 32)
  y_pred= sin_component * gauss_component
  return y_pred

# gabor gradient
def gabor_deriv_phi0(x,y,theta):
    theta0 = theta[0]
    theta1 = theta[1]
    x = 0.06 * theta1 * x + theta0
    y = y
    cos_component = torch.cos(x)
    sin_component = torch.sin(x)
    gauss_component = torch.exp(-0.5 * x *x / 16)
    deriv = cos_component * gauss_component - sin_component * gauss_component * x / 16
    deriv = 2* deriv * (sin_component * gauss_component - y)
    return torch.sum(deriv)

def gabor_deriv_phi1(x, y,theta):
    theta0 = theta[0]
    theta1 = theta[1]
    x = 0.06 * theta1 * x + theta0
    y = y
    cos_component = torch.cos(x)
    sin_component = torch.sin(x)
    gauss_component = torch.exp(-0.5 * x *x / 16)
    deriv = 0.06 * x * cos_component * gauss_component - 0.06 * x*sin_component * gauss_component * x / 16
    deriv = 2*deriv * (sin_component * gauss_component - y)
    return torch.sum(deriv)

def gabor_gradient(x, y, theta):
    dl_dphi0 = gabor_deriv_phi0(x, y, theta)
    dl_dphi1 = gabor_deriv_phi1(x, y, theta)
    # Return the gradient
    deriv =  torch.tensor([[dl_dphi0,dl_dphi1]]).T
    return deriv


