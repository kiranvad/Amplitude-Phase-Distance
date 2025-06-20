"""This file contains code to compute Amplitude-Phase distance in pytorch.

requires installation of `funcshape` code from : https://github.com/kiranvad/funcshape
"""

from typing import List, Union

import matplotlib.pyplot as plt
import torch
from funcshape.functions import (
    SRSF,
    Function,
    FunctionDistance,
    L2Metric,
    PalaisMetric,
    SineSeries,
)
from funcshape.logging import Logger
from funcshape.networks import CurveReparametrizer
from funcshape.reparametrize import reparametrize

from .geometry import SquareRootSlopeFramework


def torch_amplitude_distance(f1: Function, f2: Function, warping: Function) -> torch.Tensor:
    """Compute Phase distance between two functions given warping

    Parameters:
    ===========
        f1, f2 : funcshape.functions.Function
            function objects as defined in the Function class
        warping : funcshape.functions.Function
            Warping function aligning f2 to f1

    Returns:
    ===========
        dist : torch.Tensor
            Distance between the functions in the amplitude space
    """
    q1, q2 = SRSF(f1), SRSF(f2)
    delta = q1.qx - q2.qx
    if delta.sum().item() == 0.0:
        print("Functions are idential...")
        dist = torch.tensor(0.0)
    else:
        coordinates = (warping.x,)
        gam_dev = torch.gradient(warping.fx.squeeze(), spacing=coordinates)[0].abs()
        q_gamma = q2(warping.fx)
        y = (q1.qx.squeeze() - (q_gamma.squeeze() * torch.sqrt(gam_dev).squeeze())) ** 2
        integral = torch.trapezoid(y, q1.x)
        dist = torch.sqrt(integral)

    return dist.to(f1.fx.device)


def torch_phase_distance(f1: Function, f2: Function, warping: Function) -> torch.Tensor:
    """Compute Phase distance between two functions given warping

    Parameters:
    ===========
        f1, f2 : funcshape.functions.Function
            function objects as defined in the Function class
        warping : funcshape.functions.Function
            Warping function aligning f2 to f1

    Returns:
    ===========
        dist : torch.Tensor
            Distance between the functions in the phase space
    """
    delta = f1.fx - f2.fx
    if delta.sum() == 0:
        dist = torch.tensor(0.0)
    else:
        coordinates = (warping.x,)
        gam_dev = torch.gradient(warping.fx.squeeze(), spacing=coordinates)[0].abs()
        integrand = torch.sqrt(gam_dev).squeeze()
        theta = torch.trapezoid(integrand, x=warping.x)
        dist = torch.arccos(torch.clamp(theta, -1, 1))

    return dist.to(f1.fx.device)


def _get_warping_function_gpu(
    f1: Function,
    f2: Function,
    device: torch.device,
    **kwargs,
) -> tuple[Function, torch.nn.Module, list]:
    """GPU-enabled version of ``funcshape.functions.get_warping_function``.

    This re-implementation mirrors the original helper but moves the neural
    network and loss computation to ``device`` for faster optimization.
    """
    q1, q2 = SRSF(f1), SRSF(f2)

    n_domain = kwargs.get("n_domain", 100)
    domain_type = kwargs.get("domain_type", "linear")
    loss_func = FunctionDistance(q1, q2, k=n_domain, sample_type=domain_type)
    loss_func.to(device)

    best_error_value = float("inf")
    best_RN = None
    best_error = []
    n_restarts = kwargs.get("n_restarts", 50)
    for i in range(n_restarts):
        basis_type = kwargs.get("basis_type", "palais")
        n_basis = kwargs.get("n_basis", 20)
        if basis_type == "sine":
            basis = SineSeries(n_basis)
        elif basis_type == "L2":
            basis = L2Metric(n_basis)
        elif basis_type == "palais":
            basis = PalaisMetric(n_basis)
        else:
            raise RuntimeError(f"Basis type {basis_type} is not recognised. Should be one of [sine, L2, palais]")
        basis.to(device)

        RN = CurveReparametrizer([basis for _ in range(kwargs.get("n_layers", 15))])
        RN.to(device)

        n_iters = kwargs.get("n_iters", 100)
        optimizer = torch.optim.LBFGS(
            RN.parameters(),
            lr=kwargs.get("lr", 1e-1),
            max_iter=n_iters,
            max_eval=3 * n_iters,
            history_size=n_iters,
            line_search_fn="strong_wolfe",
        )

        error = reparametrize(
            RN,
            loss_func,
            optimizer,
            n_iters,
            Logger(0),
        )

        if error[-1] < best_error_value:
            best_error = error
            best_error_value = best_error[-1]
            best_RN = RN
            if kwargs.get("verbose", False):
                print(
                    f"Current best error : {best_error_value:.2e} at iteration {i+1} of {n_restarts}",
                    end="\r",
                )

        if best_error_value < kwargs.get("eps", 1e-2):
            if kwargs.get("verbose", False):
                print()
                print("Error threhsold reached")
            break

    if kwargs.get("verbose", False):
        print(f"Final best error : {best_error_value:.2e}")

    with torch.no_grad():
        best_RN.detach()
        x = loss_func.create_point_collection(k=n_domain).to(device)
        y = best_RN(x)

    return Function(x.squeeze(), y), best_RN, best_error


def TorchAmplitudePhaseDistance(
    t: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    *,
    device: torch.device | None = None,
    use_numpy_init: bool = False,
    **kwargs,
) -> Union[torch.Tensor, torch.Tensor, List]:
    """Compute Amplitude-Phase distance between two functions.

    If ``use_numpy_init`` is ``True`` the warping function is first computed
    using the faster NumPy implementation and that result is used directly
    to evaluate the PyTorch distance. This avoids running the slower
    gradient-descent optimization.

    Parameters:
    ===========
        t : numpy array of shape (n_domain, )
            Discrete sampling of the domain maped to [0,1]
        f1, f2 : numpy array of shape (n_domain, )
            Query and target one-dimensional functions

        kwargs : optional arguments for `get_warping_function` function.
            See get_warping_function in funcshape package for more details


    Returns:
    ===========
        da : float
            Amplitude distance between the functions
        dp : float
            Phase distance between the functions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = t.to(device)
    f1 = f1.to(device)
    f2 = f2.to(device)

    f1_fn = Function(t, f1.reshape(-1, 1))
    f2_fn = Function(t, f2.reshape(-1, 1))

    if use_numpy_init:
        # Warm start using fast NumPy dynamic programming
        time_np = t.detach().cpu().numpy()
        f1_np = f1.detach().cpu().numpy()
        f2_np = f2.detach().cpu().numpy()
        srsf = SquareRootSlopeFramework(time_np)
        q1 = srsf.to_srsf(f1_np)
        q2 = srsf.to_srsf(f2_np)
        gamma_np = srsf.get_gamma(q1, q2, **kwargs)
        gamma_t = torch.as_tensor(gamma_np, dtype=t.dtype, device=t.device)
        warping = Function(t, gamma_t.reshape(-1, 1))
        output = (warping, None, [])
    else:
        with torch.no_grad():
            output = _get_warping_function_gpu(f1_fn, f2_fn, device=device, **kwargs)

    dp = torch_phase_distance(f1_fn, f2_fn, output[0])
    da = torch_amplitude_distance(f1_fn, f2_fn, output[0])

    return da, dp, output


def plot_warping(x, f1, f2, output):
    t2x = lambda t, x: (t * (max(x) - min(x))) + min(x)
    warping = output[0]
    fig, axs = plt.subplots(1, 3, figsize=(4 * 3, 4))

    axs[0].plot(x, f1, label="ref", color="tab:blue")
    axs[0].plot(x, f2, label="query", color="tab:orange", ls="--")
    time = torch.from_numpy((x - min(x)) / (max(x) - min(x)))
    f2_ = Function(time, torch.from_numpy(f2).reshape(-1, 1))
    axs[0].plot(x, f2_(warping.fx).squeeze(), color="tab:orange", label="aligned-query")
    axs[0].legend()
    axs[0].set(xlabel="x", ylabel="f(x)", title="Functions")

    axs[1].plot(x, x, label="Identity", color="k", ls="--")
    axs[1].plot(x, t2x(warping.fx, x), color="k", label="Warping")
    axs[1].set(xlabel="x", ylabel="x", title="Warping")
    axs[1].legend()

    axs[2].semilogy(output[-1])
    axs[2].set(xlabel="Iterations", ylabel="Error")

    plt.tight_layout()
    plt.show()
