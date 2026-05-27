"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

plotting.py: visualisations for each exercise block.
Each function returns a matplotlib Figure and optionally saves or displays it.

Usage from notebook:
    import plotting

    # single tensor theta
    fig = plotting.plot_func(func=linear_scalar, x_vals=x_scalar, theta=theta)

    # list theta (unpacked as multiple arguments)
    fig = plotting.plot_func(func=shallow, x_vals=x_scalar,
                             theta=[theta_hidden, theta_out], label="shallow")

    # overlay multiple functions on the same axes
    fig, ax = plt.subplots()
    plotting.plot_func(func=relu_unit, x_vals=x_scalar, theta=theta1, ax=ax)
    plotting.plot_func(func=relu_unit, x_vals=x_scalar, theta=theta2, ax=ax,
                       title="ReLU units", save_path='figures/relu_unit')
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def plot_func(func, theta=None, label=None, title=None, ax=None, show=False, save_path=None):
    """Plot a scalar function over a range of x values.

    Can draw onto an existing Axes (for overlaying multiple functions) or
    create a new Figure if none is provided.

    Args:
        func:      callable — f(x), f(x, theta), or f(x, *theta) for multi-arg functions
        theta:     tensor for single-argument functions, list of tensors for
                   multi-argument functions (unpacked via *theta), or None
        label:     legend label — if None, a default is generated
        title:     axes title — if None, no title is set
        ax:        existing matplotlib Axes to draw onto — if None, a new figure is created
        show:      if True, display the plot interactively
        save_path: if given, save to this path (.png appended if missing)

    Returns:
        The matplotlib Figure.
    """
    x_vals = [x * 0.01 for x in range(-200, 200)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    if theta is not None:
        if isinstance(theta, list):
            y_vals = [func(x, *theta) for x in x_vals]
            if label is None:
                label = func.__name__
        else:
            y_vals = [func(x, theta) for x in x_vals]
            if label is None:
                label = rf"$\theta = ({theta[0].item():.2f}, {theta[1].item():.2f})$"
    else:
        y_vals = [func(x) for x in x_vals]
        if label is None:
            label = func.__name__

    ax.plot(x_vals, y_vals, label=label)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        if not save_path.endswith(".png"):
            save_path = save_path + ".png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_surface(func, theta, title=None, show=False, save_path=None):
    """Plot a scalar function over a 2D input grid as a 3D surface.

    Args:
        func:      callable — f(x, theta) or f(x, *theta) if theta is a list
        theta:     tensor or list of tensors
        title:     axes title — if None, no title is set
        show:      if True, display the plot interactively
        save_path: if given, save to this path (.png appended if missing)

    Returns:
        The matplotlib Figure.
    """
    grid = torch.linspace(-2, 2, 60)
    X1, X2 = torch.meshgrid(grid, grid, indexing="ij")
    
    Z = torch.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = torch.tensor([X1[i, j].item(), X2[i, j].item()])
            if isinstance(theta, list):
                Z[i, j] = func(x, *theta).item()
            else:
                Z[i, j] = func(x, theta).item()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1.numpy(), X2.numpy(), Z.numpy(), cmap="viridis", alpha=0.85)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        if not save_path.endswith(".png"):
            save_path = save_path + ".png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_timing(times, labels, N, title=None, show=False, save_path=None):
    """Bar chart comparing wall-clock times of different implementations.

    Args:
        times:     list of elapsed times in seconds
        labels:    list of labels for each bar
        N:         number of samples used in the timing experiment
        title:     axes title — if None, a default is generated
        show:      if True, display the plot interactively
        save_path: if given, save to this path (.png appended if missing)

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, times, color=["steelblue", "darkorange", "seagreen"][:len(times)])
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                t + max(times) * 0.01,
                f"{t:.4f} s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title if title else f"Timing comparison (N={N:,} samples)")
    fig.tight_layout()

    if save_path is not None:
        if not save_path.endswith(".png"):
            save_path = save_path + ".png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig