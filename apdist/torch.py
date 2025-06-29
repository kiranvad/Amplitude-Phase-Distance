"""This file contains code to compute Amplitude-Phase distance in pytorch.

requires installation of `funcshape` code from : https://github.com/kiranvad/funcshape
"""

from funcshape.functions import Function, SRSF, get_warping_function
import torch 
import matplotlib.pyplot as plt
from typing import List, Union

def torch_amplitude_distance(f1 : Function, f2 : Function, warping : Function)->torch.Tensor:
    """ Compute Phase distance between two functions given warping
    
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
    delta = q1.qx-q2.qx
    if delta.sum().item() == 0.0:
        print("Functions are idential...")
        dist = torch.tensor(0.0)
    else:
        coordinates = (warping.x, )
        gam_dev = torch.gradient(warping.fx.squeeze(), spacing=coordinates)[0].abs()
        q_gamma = q2(warping.fx)
        y = (q1.qx.squeeze() - (q_gamma.squeeze() * torch.sqrt(gam_dev).squeeze())) ** 2
        integral = torch.trapezoid(y, q1.x)
        dist = torch.sqrt(integral)
        
    return dist.to(f1.fx.device)

def torch_phase_distance(f1 : Function, f2 : Function, warping : Function)->torch.Tensor:
    """ Compute Phase distance between two functions given warping
    
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
    delta = f1.fx-f2.fx
    if delta.sum() == 0:
        dist = torch.tensor(0.0)
    else:
        coordinates = (warping.x, )
        gam_dev = torch.gradient(warping.fx.squeeze(), spacing=coordinates)[0].abs()
        integrand = torch.sqrt(gam_dev).squeeze()
        theta = torch.trapezoid(integrand, x=warping.x)
        dist = torch.arccos(torch.clamp(theta, -1, 1))    
        
    return dist.to(f1.fx.device)
    
def TorchAmplitudePhaseDistance(t : torch.Tensor, 
                           f1: torch.Tensor, 
                           f2: torch.Tensor, 
                           **kwargs)->Union[torch.Tensor, torch.Tensor, List]:
    """ Compute Amplitude-Phase distance between two functions
    
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
    f1 = Function(t, f1.reshape(-1,1))
    f2 = Function(t, f2.reshape(-1,1))
    with torch.no_grad():
        output = get_warping_function(f1, f2, **kwargs)           

    dp = torch_phase_distance(f1, f2, output[0])
    da = torch_amplitude_distance(f1, f2, output[0])

    return da, dp, output

def plot_warping(x, f1, f2, output):
    t2x = lambda t, x : (t*(max(x)-min(x))) + min(x)
    warping = output[0]
    fig, axs =plt.subplots(1,3, figsize=(4*3, 4))

    axs[0].plot(x, f1, label="ref", color="tab:blue")
    axs[0].plot(x, f2, label="query", color="tab:orange", ls="--")
    time = torch.from_numpy((x-min(x))/(max(x)-min(x)))
    f2_ = Function(time, torch.from_numpy(f2).reshape(-1,1))
    axs[0].plot(x, f2_(warping.fx).squeeze(), color="tab:orange", label="aligned-query")
    axs[0].legend()
    axs[0].set(xlabel="x", ylabel="f(x)", title="Functions")

    axs[1].plot(x, x, label="Identity", color="k", ls='--')
    axs[1].plot(x, t2x(warping.fx, x), color="k", label="Warping")
    axs[1].set(xlabel="x", ylabel="x", title="Warping")
    axs[1].legend()

    axs[2].semilogy(output[-1])
    axs[2].set(xlabel="Iterations", ylabel="Error")

    plt.tight_layout()
    plt.show()   