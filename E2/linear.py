"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

linear.py: implement all functions in this file.
Run the notebook cells to test your implementations using checks.py.
"""

import torch


# ==============================================================================
# Block 1 – Scalar linear function and scalar ReLU
# ==============================================================================

def linear_scalar(x, theta):
    """Scalar linear function: f(x) = theta_1 * x + theta_0.

    Args:
        x:     scalar input (Python float)
        theta: 1-D tensor of shape (2,) — (theta_0, theta_1)

    Returns:
        Scalar output theta_1 * x + theta_0.
    """
    return (theta[1] * x) + theta[0]


def relu_scalar(x):
    """Scalar ReLU: relu(x) = max(0, x).

    Args:
        x: scalar input (Python float)

    Returns:
        max(0, x)
    """
    return max(0, x)


# ==============================================================================
# Block 2 – Shallow neural network (scalar input)
# ==============================================================================

def relu_unit(x, theta):
    """Single ReLU unit: sigma(theta_1 * x + theta_0).

    Args:
        x:     scalar input (Python float)
        theta: 1-D tensor of shape (2,) — (theta_0, theta_1)

    Returns:
        Scalar output.
    """
    return relu_scalar(linear_scalar(x,theta))


def shallow(x, theta_hidden, theta_out):
    """Shallow neural network with scalar input and scalar output.

    Computes: theta_out[0] + sum_j theta_out[j] * relu_unit(x, theta_hidden[j-1])

    Args:
        x:            scalar input (Python float)
        theta_hidden: list of k tensors of shape (2,), one per hidden unit
                      — each is (theta_j0, theta_j1)
        theta_out:    1-D tensor of shape (k+1,)
                      — theta_out[0] is bias, theta_out[j] is weight for unit j

    Returns:
        Scalar output.
    """
    sum = 0
    for index, theta in enumerate(theta_hidden):
        sum += theta_out[index+1] * relu_unit(x, theta)
    return theta_out[0] + sum
    


# ==============================================================================
# Block 3 – Vector input, scalar output
# ==============================================================================

def linear_vector(x, theta):
    """Linear function with vector input and scalar output.

    Computes: theta_1 · x + theta_0, where theta_1 = theta[1:].

    Args:
        x:     1-D tensor, shape (d,)
        theta: 1-D tensor, shape (d+1,) — (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Scalar output (0-d tensor).
    """
    return (theta[1:] @ x) + theta[0]


def relu_tensor(x):
    """Element-wise ReLU for a tensor of any shape.

    Args:
        x: tensor

    Returns:
        Tensor of same shape with negative values zeroed out.
    """
    return torch.relu(x)


def relu_unit_vector(x, theta):
    """Single ReLU unit with vector input: sigma(theta_1 · x + theta_0).

    Args:
        x:     1-D tensor, shape (d,)
        theta: 1-D tensor, shape (d+1,) — (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Scalar output (0-d tensor).
    """
    return relu_tensor(linear_vector(x, theta))


def shallow_vector(x, theta_hidden, theta_out):
    """Shallow network with vector input and scalar output.

    Computes: theta_out[0] + sum_j theta_out[j] * relu_unit_vector(x, theta_hidden[j-1])

    Args:
        x:            1-D tensor, shape (d,)
        theta_hidden: list of k tensors of shape (d+1,), one per hidden unit
        theta_out:    1-D tensor of shape (k+1,)

    Returns:
        Scalar output (0-d tensor).
    """
    total = theta_out[0].item()
    for index, theta in enumerate(theta_hidden):
            total += theta_out[index+1] * relu_unit_vector(x, theta)
    return total


# ==============================================================================
# Block 4 – Batching
# ==============================================================================

def shallow_batch_loop(X, theta_hidden, theta_out):
    """Shallow network forward pass over a batch using a Python loop over samples.

    Args:
        X:            2-D tensor, shape (N, d)
        theta_hidden: list of k tensors of shape (d+1,), one per hidden unit
        theta_out:    1-D tensor of shape (k+1,)

    Returns:
        Output tensor, shape (N,)
    """
    result = []
    for x_i in X:
        result.append(shallow_vector(x_i, theta_hidden,theta_out))
    return torch.stack(result)
    


def linear_batch(X, theta):
    """Linear function over a batch of samples with scalar output per sample.

    Vectorised version of linear_vector over N samples.

    Args:
        X:     2-D tensor, shape (N, d)
        theta: 1-D tensor, shape (d+1,) — (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Output tensor, shape (N,)
    """
    return X @ theta[1:] + theta[0]  # X here is metrix and it directly multiply with the theta so it like we are doing in book


def shallow_batch(X, theta_hidden, theta_out):
    """Shallow network forward pass over a batch, vectorised over samples.

    Uses linear_batch for each hidden unit — processes all N samples at once
    per unit, but still loops over hidden units.

    Args:
        X:            2-D tensor, shape (N, d)
        theta_hidden: list of k tensors of shape (d+1,), one per hidden unit
        theta_out:    1-D tensor of shape (k+1,)

    Returns:
        Output tensor, shape (N,)
    """
    result = torch.zeros(X.shape[0]) + theta_out[0]
    for index, theta in enumerate(theta_hidden):
        result += theta_out[index+1] * relu_tensor(linear_batch(X, theta))  # It already tensor at the time of output so no need of tensor.stake() at the return statement 
    return result


def linear_layer(X, Theta):
    """Linear layer mapping a batch of inputs to multiple outputs.

    Applies a linear transformation to all N samples simultaneously,
    producing k outputs per sample.

    Args:
        X:     2-D tensor, shape (N, d)
        Theta: 2-D tensor, shape (k, d+1) — each row is one output unit's parameters
               (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Output tensor, shape (N, k)

    J.J.
    Good thinking, but linear_batch won't work directly here because it only handles one theta vector at a time, 
    giving you (N,). Here you need to do it for k units at once, giving (N, k).
    Think about it this way — Theta has shape (k, d+1), so:

    Theta[:, 0] → all k biases, shape (k,)
    Theta[:, 1:] → all k weight vectors, shape (k, d)

    Now think about what matrix operation on X of shape (N, d) and Theta[:, 1:] of shape (k, d) gives you shape (N, k).
    Hint — you need to transpose one of them. Ask yourself:
    (N, d) @ (d, k) → (N, k)   ✓
    So Theta[:, 1:] needs to be transposed. In PyTorch that's .T.
    """
    return X @ Theta[:, 1:].T + Theta[:, 0]

def shallow_batch_vectorised(X, Theta_hidden, theta_out):
    """Shallow network forward pass, fully vectorised over samples and hidden units.

    Uses linear_layer to compute all hidden activations at once.

    Args:
        X:             2-D tensor, shape (N, d)
        Theta_hidden:  2-D tensor, shape (k, d+1) — each row is one hidden unit's parameters
        theta_out:     1-D tensor of shape (k+1,)

    Returns:
        Output tensor, shape (N,)
    """
    result = torch.zeros(X.shape[0]) + theta_out[0]
    result += relu_tensor(linear_layer(X, Theta_hidden)) @ theta_out[1:]
    return result


# ==============================================================================
# Block 6 – Two-layer MLP
# ==============================================================================

def mlp_batch(X, Theta_1, Theta_2, theta_out):
    """Two-layer MLP forward pass, fully vectorised.

    Args:
        X:         2-D tensor, shape (N, d)
        Theta_1:   2-D tensor, shape (k1, d+1)  — first hidden layer parameters
        Theta_2:   2-D tensor, shape (k2, k1+1) — second hidden layer parameters
        theta_out: 1-D tensor, shape (k2+1,)    — output layer parameters

    Returns:
        Output tensor, shape (N,)
    """
    pass
