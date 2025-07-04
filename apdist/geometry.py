import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz

class SquareRootSlopeFramework:
    """Square Root Slope Framework (SRSF)
    
    Parameters:
    ===========
        time : numpy array of shape (n_domain, )
            Discrete mapping of domain into [0,1]
            
    Attributes:
    ===========
        to_srsf : Compute SRSF of a function
        from_srsf : Compute the function from SRSF
        warp_f_gamma : Apply warping to a function
        warp_q_gamma : Apply warping to SRSF of a function
        get_gamma : Compute warping function given two SRSF
    """
    def __init__(self, time):
        self.time = time 

    def to_srsf(self, f):
        """Compute SRSF of a function
        
        Parameters:
        ===========
            f : numpy array of shape (n_domain, )
                Discrete evaluation of a function
                
        Returns:
        ========
            q : numpy array of shape (n_domain, )
                Discrete SRSF evaluation of a function            
        """
        spl = UnivariateSpline(self.time, f, s=0)
        grad = spl.derivative(n=1)(self.time)
        q = grad / np.sqrt(np.fabs(grad) + 1e-3)

        return q

    def from_srsf(self, q, f0=0.0):
        """Reconstruct function from SRSF
        
        Parameters:
        ===========
            q : numpy array of shape (n_domain, )
                Discrete SRSF evaluation of a function  
            f0 : float
                Initial shift of a function (default, 0.0) 
                
        Returns:
        ========
            f : numpy array of shape (n_domain, )
                Discrete evaluation of a function            
        """        
        integrand = q*np.fabs(q)
        f = f0 + cumtrapz(integrand,self.time,initial=0)
        
        return f

    def warp_f_gamma(self, f, gam):
        """Warp a function f with a gamma function
        
        Parameters:
        ===========
            f : numpy array of shape (n_domain, )
                Discrete evaluation of a function  
            gam : numpy array of shape (n_domain, )
                Gamma function to warp f with 
                
        Returns:
        ========
            f_gamma : numpy array of shape (n_domain, )
                Warped function 'f' with 'gam'         
        """ 
        f_gamma = np.interp(gam, self.time, f)

        return f_gamma
        
    def warp_q_gamma(self, q, gam):
        """Warp a function q with a gamma function
        
        Parameters:
        ===========
            q : numpy array of shape (n_domain, )
                Discrete evaluation of a function  
            gam : numpy array of shape (n_domain, )
                Gamma function to warp f with 
                
        Returns:
        ========
            q_gamma : numpy array of shape (n_domain, )
                Warped function 'q' with 'gam'         
        """ 
        gam_dev = np.gradient(gam, self.time)
        tmp = np.interp(gam, self.time, q)

        q_gamma = tmp * np.sqrt(gam_dev)

        return q_gamma
        
    def get_gamma(self, q1, q2, **kwargs):
        """Compute warping function given two SRSFs

        Parameters:
        ===========
            q1, q2 : numpy array of shape (n_domain, )
                Discrete SRSF evaluation of pair of functions
            lam : float
                Regularization parameter of Dynamic Programming algorithm


        Returns:
        ========
            gamma : numpy array of shape (n_domain, )
                Warping function to align 'q2' with 'q1'


        This function is a heavylift from the python 'fdasrsf' package.
        See https://github.com/jdtuck/fdasrsf_python for more details.
        """
        optim = kwargs.pop("optim", "DP")
        if optim == "DP":
            try:
                import optimum_reparamN2 as orN2

                gamma = orN2.coptimum_reparam(np.ascontiguousarray(q1),
                                            self.time,
                                            np.ascontiguousarray(q2),
                                            kwargs.pop("lambda", 0.0),
                                            kwargs.pop("grid_dim", 7)
                                            )
            except ImportError:
                # Fallback to identity warping if optimum_reparamN2 is not available
                import warnings
                warnings.warn("optimum_reparamN2 not available. Using identity warping. "
                             "Install the warping package for full functionality: "
                             "pip install git+https://github.com/kiranvad/warping.git")
                gamma = self.time.copy()
        else:
            raise RuntimeError("Method %s for gamma optimization is not recognized"%optim)

        gamma = (self.time[-1] - self.time[0]) * gamma + self.time[0]

        return gamma

class WarpingManifold:
    """Manifold for the space of one-dimensional warping functions
    
    Parameters:
    ===========
        time : numpy array of shape (n_domain, )
            Discrete mapping of domain into [0,1]
            
    Attributes:
    ===========
        inner_product : Compute inner product between tangent vector of a function space
        norm : Compute the norm of a tangent vector of a function space
        log : Apply logarthim function of warping manifold
        exp : Apply exponential function of warping manifold
        inverse : Compute inverse of a warping function
        center : Compute center of a set of warping functions        
    """
    def __init__(self, time):
        self.time = time
    
    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        # base_point parameter kept for API compatibility
        _ = base_point  # Suppress unused parameter warning
        ip = np.trapz(tangent_vec_a*tangent_vec_b, self.time)

        return ip

    def norm(self, tangent_vec, base_point=None):
        # base_point parameter kept for API compatibility
        _ = base_point  # Suppress unused parameter warning
        l2norm = np.sqrt(self.inner_product(tangent_vec, tangent_vec))

        return l2norm
    
    def log(self, base_point, point):
        tmp = self.inner_product(base_point, point)
        if tmp > 1:
            tmp = 1
        if tmp < -1:
            tmp = -1

        theta = np.arccos(tmp)

        if (theta < 1e-10):
            exp_inv = np.zeros(point.shape[0])
        else: 
            exp_inv = theta / np.sin(theta) * (point - np.cos(theta)*base_point)

        return exp_inv, theta
    
    def exp(self, point, base_point):
        norm = self.norm(base_point)
        if norm.sum() == 0:
            expgam = np.cos(norm) * point
        else:
            expgam = np.cos(norm) * point + np.sin(norm) * base_point / norm

        return expgam


    def inverse(self, gam):
        N = gam.size
        x = np.linspace(0,1,N)
        s = interp1d(gam, x)
        gamI = s(x)
        gamI = (gamI - gamI[0]) / (gamI[-1] - gamI[0])
        
        return gamI

    def center(self, gam):
        if gam.ndim > 1:
            T, n = gam.shape
        else:
            return self.inverse(gam)

        psi = np.zeros_like(gam)
        for k in range(0, n):
            psi[:, k] = np.sqrt(np.gradient(gam[:, k], self.time))

        # Find Direction
        mnpsi = psi.mean(axis=1)
        a = mnpsi.repeat(n)
        d1 = a.reshape(T, n)
        d = (psi - d1) ** 2
        dqq = np.sqrt(d.sum(axis=0))
        min_ind = dqq.argmin()
        mu = psi[:, min_ind]

        maxiter = 501

        stp = 0.3
        itr = 0

        error = np.inf

        while (error > 1e-6) and (itr < maxiter):
            vec = np.zeros((T, n))
            for i in range(n):
                out, _ = self.log(mu, psi[:, i])  # theta not used in this context
                vec[:, i] = out

            vbar = vec.mean(axis=1)
            error = self.norm(vbar)

            mu = self.exp(mu, stp * vbar)
            itr += 1

        gam_mu = cumtrapz(mu * mu, self.time, initial=0)
        gam_mu = (gam_mu - gam_mu.min()) / (gam_mu.max() - gam_mu.min())
        gamI = self.inverse(gam_mu)

        return gamI