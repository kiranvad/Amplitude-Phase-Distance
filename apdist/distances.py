import numpy as np

from .geometry import *

def _grad(f, binsize):
    n = f.shape[0]
    g = np.zeros(n)
    h = binsize*np.arange(1,n+1)
    g[0] = (f[1] - f[0])/(h[1]-h[0])
    g[-1] = (f[-1] - f[(-2)])/(h[-1]-h[-2])

    h = h[2:]-h[0:-2]
    g[1:-1] = (f[2:]-f[0:-2])/h[0]

    return g

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
        M = len(time)
        gam_dev = _grad(gam, 1 / np.double(M - 1))
        tmp = np.interp((time[-1] - time[0]) * gam + time[0], time, q2)

        qw = tmp * np.sqrt(gam_dev)

        y = (qw - q1) ** 2
        tmp = np.diff(time)*(y[0:-1]+y[1:])/2
        dist = np.sqrt(tmp.sum())
       
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
        M = len(time)
        gam_dev = _grad(gam, 1 / np.double(M - 1))
        theta = np.trapz(np.sqrt(gam_dev),x=time)
        if theta > 1:
            theta = 1
        elif theta < -1:
            theta = -1
            
        dist = np.arccos(theta)    
        
    return dist
    
def AmplitudePhaseDistance(x, f1, f2, **kwargs):
    """ Compute Amplitude-Phase distance between two functions
    
    Parameters:
    ===========
        x : numpy array of shape (n_domain, )
            Discrete sampling of the domain    
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
    time = x-min(x)/(max(x)-min(x))
    SRSF = SquareRootSlopeFramework(time)
    q1 = SRSF.to_srsf(f1)
    q2 = SRSF.to_srsf(f2)
    gam = SRSF.get_gamma(q1, q2, **kwargs)            
    gam = (gam - gam[0]) / (gam[-1] - gam[0])
    
    dp = _phase_distance(time, q1, q2, gam)
    da = _amplitude_distance(time, q1, q2, gam)

    return da, dp