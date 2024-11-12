import numpy as np 

# Pseudo-Voigt function definition with multiple components
def multi_pseudo_voigt(r, A1, mu1, sigma1, gamma1, eta1, A2, mu2, sigma2, gamma2, eta2, A3, mu3, sigma3, gamma3, eta3):
    """
    Multi-component Pseudo-Voigt function to model background with multiple peaks.
    
    Parameters:
    - r: Radial distance array.
    - A1, mu1, sigma1, gamma1, eta1: Parameters for the first Pseudo-Voigt component.
    - A2, mu2, sigma2, gamma2, eta2: Parameters for the second Pseudo-Voigt component.
    - A3, mu3, sigma3, gamma3, eta3: Parameters for the third Pseudo-Voigt component.
    """
    gaussian1 = np.exp(-((r - mu1) ** 2) / (2 * sigma1 ** 2))
    lorentzian1 = gamma1 ** 2 / ((r - mu1) ** 2 + gamma1 ** 2)
    pv1 = A1 * (eta1 * lorentzian1 + (1 - eta1) * gaussian1)
    
    gaussian2 = np.exp(-((r - mu2) ** 2) / (2 * sigma2 ** 2))
    lorentzian2 = gamma2 ** 2 / ((r - mu2) ** 2 + gamma2 ** 2)
    pv2 = A2 * (eta2 * lorentzian2 + (1 - eta2) * gaussian2)
    
    gaussian3 = np.exp(-((r - mu3) ** 2) / (2 * sigma3 ** 2))
    lorentzian3 = gamma3 ** 2 / ((r - mu3) ** 2 + gamma3 ** 2)
    pv3 = A3 * (eta3 * lorentzian3 + (1 - eta3) * gaussian3)
    
    return pv1 + pv2 + pv3
