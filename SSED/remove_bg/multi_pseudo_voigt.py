import numpy as np 

# General Pseudo-Voigt function definition with multiple components
def multi_pseudo_voigt(r, *params):
    """
    Multi-component Pseudo-Voigt function to model background with multiple peaks.
    
    Parameters:
    - r: Radial distance array.
    - params: Flattened list of parameters for each component (A, mu, sigma, gamma, eta).
    
    Returns:
    - The sum of all Pseudo-Voigt components.
    """
    num_components = len(params) // 5
    total_pv = np.zeros_like(r)
    
    for i in range(num_components):
        A = params[i * 5]
        mu = params[i * 5 + 1]
        sigma = params[i * 5 + 2]
        gamma = params[i * 5 + 3]
        eta = params[i * 5 + 4]
        
        gaussian = np.exp(-((r - mu) ** 2) / (2 * sigma ** 2))
        lorentzian = gamma ** 2 / ((r - mu) ** 2 + gamma ** 2)
        pv = A * (eta * lorentzian + (1 - eta) * gaussian)
        
        total_pv += pv
    
    return total_pv
