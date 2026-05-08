"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

checks.py: sanity checks for each exercise block.
Call check_block1(), check_block2(), etc. from the notebook after
implementing the corresponding functions in linear.py.
"""

import torch
from linear import linear_scalar, relu_scalar, relu_unit, shallow, linear_vector, relu_tensor


def check_block1():
    """Sanity checks for linear_scalar and relu_scalar."""
    theta = torch.tensor([-1.0, 3.0])
    assert linear_scalar(2.0, theta) == 5.0, "linear_scalar implementation not correct"
    assert relu_scalar(2.5) == 2.5,           "relu_scalar implementation not correct"
    print("Block 1 checks passed.")


def check_block2():
    """Sanity checks for relu_unit and shallow."""
    th = torch.tensor([-0.5, 2.0])
    assert relu_unit(1.0, th)  == 1.5, "relu_unit implementation not correct"
    assert relu_unit(-1.0, th) == 0.0, "relu_unit implementation not correct"

    th1 = torch.tensor([-0.5, 2.0])
    th2 = torch.tensor([1.0, -1.5])
    theta_out = torch.tensor([0.5, 1.0, -1.0])
    assert shallow(1.0, [th1, th2], theta_out) == 2.0, "shallow implementation not correct"
    print("Block 2 checks passed.")


def check_block3():
    """Sanity checks for linear_vector and relu_tensor."""
    x = torch.tensor([1.0, 2.0])
    theta = torch.tensor([0.3, 0.5, -1.0])
    assert abs(linear_vector(x, theta) - (-1.2)) < 1e-5, "linear_vector implementation not correct"
    t = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.allclose(relu_tensor(t), torch.tensor([0.0, 0.0, 2.0])), "relu_tensor implementation not correct"
    print("Block 3 checks passed.")


def check_block4_1():
    """Sanity check for shallow_batch_loop."""
    from linear import shallow_batch_loop
    theta_hidden = [torch.tensor([0.0, 1.0, 0.5]), torch.tensor([0.5, -1.0, 1.0])]
    theta_out = torch.tensor([0.0, 1.0, -1.0])
    X = torch.tensor([[1.0, 2.0], [0.5, -1.0], [0.0, 0.0]])
    out = shallow_batch_loop(X, theta_hidden, theta_out)
    assert out.shape == torch.Size([3]), "shallow_batch_loop output shape not correct"
    assert torch.allclose(out, torch.tensor([0.5, 0.0, -0.5]), atol=1e-5), "shallow_batch_loop implementation not correct"
    print("Block 4.1 checks passed.")


def check_block4_3():
    """Sanity checks for linear_batch and shallow_batch."""
    from linear import linear_batch, shallow_batch_loop, shallow_batch
    X = torch.tensor([[1.0, 2.0], [0.5, -1.0], [0.0, 0.0]])
    theta_hidden = [torch.tensor([0.0, 1.0, 0.5]), torch.tensor([0.5, -1.0, 1.0])]
    theta_out = torch.tensor([0.0, 1.0, -1.0])

    # linear_batch
    th = torch.tensor([0.0, 1.0, 0.5])
    out_lb = linear_batch(X, th)
    assert out_lb.shape == torch.Size([3]), "linear_batch output shape not correct"
    assert torch.allclose(out_lb, torch.tensor([2.0, 0.0, 0.0]), atol=1e-5), "linear_batch implementation not correct"

    # shallow_batch must match loop
    out_loop  = shallow_batch_loop(X, theta_hidden, theta_out)
    out_batch = shallow_batch(X, theta_hidden, theta_out)
    assert out_batch.shape == torch.Size([3]), "shallow_batch output shape not correct"
    assert torch.allclose(out_loop, out_batch, atol=1e-5), "shallow_batch implementation not correct"
    print("Block 4.3 checks passed.")


def check_block4_5():
    """Sanity checks for linear_layer and shallow_batch_vectorised."""
    from linear import linear_batch, shallow_batch_loop, linear_layer, shallow_batch_vectorised
    X = torch.tensor([[1.0, 2.0], [0.5, -1.0], [0.0, 0.0]])
    theta_hidden = [torch.tensor([0.0, 1.0, 0.5]), torch.tensor([0.5, -1.0, 1.0])]
    Theta_hidden = torch.stack(theta_hidden)
    theta_out = torch.tensor([0.0, 1.0, -1.0])

    # linear_layer
    out_ll = linear_layer(X, Theta_hidden)
    assert out_ll.shape == torch.Size([3, 2]), "linear_layer output shape not correct"

    # shallow_batch_vectorised must match loop
    out_loop = shallow_batch_loop(X, theta_hidden, theta_out)
    out_full = shallow_batch_vectorised(X, Theta_hidden, theta_out)
    assert out_full.shape == torch.Size([3]), "shallow_batch_vectorised output shape not correct"
    assert torch.allclose(out_loop, out_full, atol=1e-5), "shallow_batch_vectorised implementation not correct"
    print("Block 4.5 checks passed.")


def check_block6():
    """Sanity check for mlp_batch."""
    from linear import mlp_batch
    torch.manual_seed(0)
    N, d, k1, k2 = 5, 3, 4, 2
    X = torch.randn(N, d)
    Theta_1   = torch.randn(k1, d + 1)
    Theta_2   = torch.randn(k2, k1 + 1)
    theta_out = torch.randn(k2 + 1)
    out = mlp_batch(X, Theta_1, Theta_2, theta_out)
    assert out.shape == torch.Size([N]), "mlp_batch output shape not correct"
    print("Block 6 checks passed.")