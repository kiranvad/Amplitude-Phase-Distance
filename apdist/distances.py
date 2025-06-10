import numpy as np
from .geometry import *

def _amplitude_distance(time, q1, q2, gam):
    """ Compute Amplitude distance between two SRSF
    
    Parameters:
    ===========
        time : numpy array of shape (n_domain, )
            Time domain samples    
        q1, q2 : numpy array of shape (n_domain, )
            SRSFs of two functions q = SRSF(f)
        gam : numpy array of shape (n_domain, )
            Warping function aligning q2 to q1  
                        
    Returns:
    ===========
        dist : float
            Distance between the SRSFs in the amplitude space
    """
    delta = q1-q2
    if delta.sum() == 0:
        dist = 0
    else:
        gam_dev = np.gradient(gam, time)
        q_gamma = np.interp(gam, time, q2)
        y = (q1 - (q_gamma * np.sqrt(gam_dev))) ** 2

        dist = np.sqrt(np.trapezoid(y, time))
       
    return dist

def _phase_distance(time, q1, q2, gam):
    """ Compute Phase distance between two SRSF
    
    Parameters:
    ===========
        time : numpy array of shape (n_domain, )
            Time domain samples    
        q1, q2 : numpy array of shape (n_domain, )
            SRSFs of two functions q = SRSF(f)
        gam : numpy array of shape (n_domain, )
            Warping function aligning q2 to q1        
            
    Returns:
    ===========
        dist : float
            Distance between the SRSFs in the amplitude space
    """
    delta = q1-q2
    if delta.sum() == 0:
        dist = 0
    else:
        gam_dev = np.gradient(gam, time)
        theta = np.trapezoid(np.sqrt(gam_dev), x=time)
        dist = np.arccos(np.clip(theta, -1, 1))    
        
    return dist
    
def AmplitudePhaseDistance(t, f1, f2, **kwargs):
    """ Compute Amplitude-Phase distance between two functions
    
    Parameters:
    ===========
        t : numpy array of shape (n_domain, )
            Discrete sampling of the domain maped to [0,1]   
        f1, f2 : numpy array of shape (n_domain, )
            Query and target one-dimensional functions 
            
        kwargs : optional arguments for `get_gamma` function.
            See geometry.SqaureRootSlopeFramework for more details
              
            
    Returns:
    ===========
        da : float
            Amplitude distance between the functions
        dp : float
            Phase distance between the functions    
    """
    SRSF = SquareRootSlopeFramework(t)
    q1 = SRSF.to_srsf(f1)
    q2 = SRSF.to_srsf(f2)
    gam = SRSF.get_gamma(q1, q2, **kwargs)            
    
    dp = _phase_distance(t, q1, q2, gam)
    da = _amplitude_distance(t, q1, q2, gam)

    return da, dp