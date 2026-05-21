"""THWS/MAI/Introduction to Deep Learning - Assignment 2 Part 2 - functions

Magda Gregorová, April 2026

Implement the functions marked with TODO.
Do not change function signatures.
Use only PyTorch tensor operations — no numpy, no torch.nn.

For MSE, the local and global gradients are always identical since MSE is the
root of the computation graph — there is no upstream gradient to multiply by.
We therefore provide a single backward function for MSE.

For Linear and ReLU, the backward pass is split into two functions:
  _lgrad: computes local gradients (partial derivatives of this node's output
          w.r.t. its inputs), independent of the rest of the graph
  _ggrad: computes global gradients by applying the chain rule with upstream
          gout, and returns the results
"""

import torch


# ==============================================================================
# Section 1 — MSE backward
# ==============================================================================

def mse_backward_scalar(y_pred, y):
    """Gradient of scalar MSE loss w.r.t. prediction.

    Since MSE is the root of the computation graph, local and global gradients
    are always identical.

    Args:
        y_pred: float - predicted value
        y:      float - true value

    Returns:
        float - gradient dL/d(y_pred)
    
    J.J. for scaler case dy/​dL​=2(y_pred​−y) there is no 1/N.
    """
    # TODO: implement
    return 2 * (y_pred - y)


def mse_backward(y_pred, y):
    """Gradient of batch MSE loss w.r.t. each prediction.

    Since MSE is the root of the computation graph, local and global gradients
    are always identical.

    Args:
        y_pred: torch.tensor of shape (N, 1)
        y:      torch.tensor of shape (N, 1)

    Returns:
        torch.tensor of shape (N, 1)

    J.J. for tensor case there is always N. so; dy/​dL​=2/N(y_pred​−y)
    """
    # TODO: implement
    N = y_pred.numel()
    return (2/N) * (y_pred - y)


# ==============================================================================
# Section 2 — Linear backward
# ==============================================================================

def linear_lgrad_scalar(x, theta):
    """Local gradients for scalar linear function z = theta_1 * x + theta_0.

    Args:
        x:     torch.tensor of shape () - scalar input
        theta: torch.tensor of shape (2,) - (theta_0, theta_1)

    Returns:
        tuple: (lgrad_theta_0, lgrad_theta_1, lgrad_x)
               lgrad_theta_0: torch.tensor of shape ()
               lgrad_theta_1: torch.tensor of shape ()
               lgrad_x:       torch.tensor of shape ()

    Hint: use .clone() when returning parts of existing tensors.

    """
    """
    J.J.
    dz/dθ_0 = 1 ; dz/dθ_1 = x; dz/dx=θ_1

    torch.tensor(1.0)   # hardcoded: we not using this right now.
    > Always creates a float32 tensor
    > Always on CPU
    > If your model is on GPU or uses float64, this could cause a device/dtype mismatch error

    theta[1].clone() * 0 + 1
    > Inherits dtype from theta automatically
    > Inherits device from theta automatically
    > So if theta is on GPU or float64, this just works

    Breaking down theta[1].clone() * 0 + 1:
    theta[1]        → grab the scalar tensor (theta_1 value, e.g. tensor(3.5))
    .clone()        → make a safe copy so we don't touch the original
    * 0             → zero it out → tensor(0.0)  (but keeps dtype & device!)
    + 1             → add 1      → tensor(1.0)
    """
    # TODO: implement
    return (torch.ones_like(theta[1]), x.clone(), theta[1].clone())


def linear_ggrad_scalar(gout, x, theta):
    """Global gradients for scalar linear function z = theta_1 * x + theta_0.

    Applies the chain rule with upstream gout.
    In the scalar case this is elementwise: gout * lgrad.

    Args:
        gout:  torch.tensor of shape () - upstream gradient dL/dz
        x:     torch.tensor of shape () - scalar input
        theta: torch.tensor of shape (2,) - (theta_0, theta_1)

    Returns:
        tuple: (ggrad_theta_0, ggrad_theta_1, ggrad_x)
               each a torch.tensor of shape ()
    """
    # TODO: implement using linear_lgrad_scalar
    lgrad_lst = linear_lgrad_scalar(x, theta)
    ggrad_lst = []

    for grd in lgrad_lst:
        ggrad_lst.append(gout * grd)

    return tuple(ggrad_lst)


def linear_lgrad(ins, theta_1, theta_0):
    """Local gradient factors for batched linear layer Z = X @ theta_1.T + theta_0.

    Args:
        ins:     torch.tensor of shape (N, in_features)
        theta_1: torch.tensor of shape (out_features, in_features)
        theta_0: torch.tensor of shape (1, out_features)

    Returns:
        tuple: (lgrad_theta_1_factor, lgrad_theta_0_factor, lgrad_ins)
               lgrad_theta_1_factor: torch.tensor of shape (N, in_features)
               lgrad_theta_0_factor: torch.tensor of shape (N, out_features)
               lgrad_ins:            torch.tensor of shape (out_features, in_features)
    """
    """
    dz/dθ_0 = 1 ; dz/dθ_1 = x; dz/dx=θ_1
    shape confusion of theta_0?
    Notice the word "factor" for theta gradients — this hints that they can't be fully computed yet, 
    they still need to be combined with gout later in linear_ggrad.

    """
    # TODO: implement
    ones = torch.ones(ins.shape[0], theta_0.shape[1])
    return (ins.clone(), ones, theta_1.clone())


def linear_ggrad(gout, ins, theta_1, theta_0):
    """Global gradients for batched linear layer Z = X @ theta_1.T + theta_0.

    Applies the chain rule with upstream gout.
    In the matrix case this involves matrix products — think carefully about
    how the dimensions work out.

    Args:
        gout:    torch.tensor of shape (N, out_features) - upstream gradient dL/dZ
        ins:     torch.tensor of shape (N, in_features)
        theta_1: torch.tensor of shape (out_features, in_features)
        theta_0: torch.tensor of shape (1, out_features)

    Returns:
        tuple: (ggrad_theta_1, ggrad_theta_0, ggrad_ins)
               ggrad_theta_1: torch.tensor of shape (out_features, in_features)
               ggrad_theta_0: torch.tensor of shape (1, out_features)
               ggrad_ins:     torch.tensor of shape (N, in_features)
    """
    # TODO: implement using linear_lgrad
    lgrd = linear_lgrad(ins, theta_1, theta_0) 
    lg_theta_1, lg_theta_0, lg_ins = lgrd

    ggrad_theta_1 = gout.T @ lg_theta_1
    ggrad_theta_0 = (gout * lg_theta_0).sum(dim=0, keepdim=True)
    ggrad_ins = gout @ theta_1

    print("lg_theta_0 shape:", lg_theta_0.shape)
    print("gout shape:", gout.shape)
    
    return (ggrad_theta_1, ggrad_theta_0, ggrad_ins)

# ==============================================================================
# Section 3 — ReLU backward
# ==============================================================================

def relu_lgrad_scalar(z):
    """Local gradient for scalar ReLU: a = relu(z) = max(0, z).

    Args:
        z: torch.tensor of shape () - scalar pre-activation

    Returns:
        torch.tensor of shape () - local gradient da/dz
    """
    """
    J.J.
    x = torch.tensor(3.14)   # scalar tensor, shape ()
    x = torch.tensor(0.0)    # scalar tensor, shape ()
    x = torch.tensor(-5)     # scalar tensor, shape ()

    another equal sol. is:
    if (z>0).float():
        return torch.ones_like(z)   # when giving z it return shape like z; shape()
    else:
        return torch.zeros_like(z)
    """
    # TODO: implement
    if z > 0:
        return torch.ones_like(z)
    else:
        return torch.zeros_like(z)


def relu_ggrad_scalar(gout, z):
    """Global gradient for scalar ReLU.

    Applies the chain rule with upstream gout.

    Args:
        gout: torch.tensor of shape () - upstream gradient dL/da
        z:    torch.tensor of shape () - scalar pre-activation

    Returns:
        torch.tensor of shape () - global gradient dL/dz
    """
    # TODO: implement using relu_lgrad_scalar
    return gout * relu_lgrad_scalar(z)


def relu_lgrad(ins):
    """Local gradient for batch ReLU: A = relu(Z), element-wise.

    Args:
        ins: torch.tensor of any shape - pre-activations Z

    Returns:
        torch.tensor of same shape - local gradient
    """
    # TODO: implement
    return (ins > 0).float() # This works element-wise on a tensor of any shape — every element gets checked independently, returning 1.0 or 0.0


def relu_ggrad(gout, ins):
    """Global gradient for batch ReLU.

    Applies the chain rule with upstream gout.

    Args:
        gout: torch.tensor of same shape as ins - upstream gradient dL/dA
        ins:  torch.tensor of any shape - pre-activations Z

    Returns:
        torch.tensor of same shape - global gradient dL/dZ
    """
    # TODO: implement using relu_lgrad
    return gout * relu_lgrad(ins)
