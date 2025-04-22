import numpy as np

def soft_clamp_weights(sigmas, sigma_floor=0.005):
    '''
    Applies a soft clamp to the sigma values used in weighting for regression.
    Ensures that no weight becomes excessively large due to underestimated sigma.
    
    Parameters:
    - sigmas: array-like of sigma values (standard deviations from fit)
    - sigma_floor: minimum allowable sigma to prevent overweighting
    
    Returns:
    - weights: 1 / max(sigma, sigma_floor)^2
    '''
    return 1.0 / np.square(np.maximum(sigmas, sigma_floor))
import warnings
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

def _spline_min(lam, flux):
    fine = np.linspace(lam.min(), lam.max(), 500)
    spline = UnivariateSpline(lam, flux - 1.0, s=0)
    smoothed = spline(fine) + 1.0
    mu = fine[np.argmin(smoothed)]
    mu_err = 0.0
    return mu, mu_err

def _voigt_profile(x, amp, mu, sigma, gamma, c):
    from scipy.special import wofz
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return c - amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def _skew_gauss(x, amp, mu, sigma, alpha, c):
    from scipy.stats import norm
    from scipy.special import erf
    t = (x - mu) / sigma
    return c - amp * norm.pdf(t) * (1 + erf(alpha * t / np.sqrt(2)))

def _pseudo_voigt_asym(x, amp, mu, sigma, gamma, eta, c):
    # Simplified asymmetric pseudo-Voigt
    gaussian = np.exp(-((x - mu)**2) / (2 * sigma**2))
    lorentz = gamma**2 / ((x - mu)**2 + gamma**2)
    return c - amp * (eta * lorentz + (1 - eta) * gaussian)

def fit_absorption_line_ladder(lam, flux, ivar=None, lam0=None):
    methods = []
    rejection = None

    window = 6 * np.median(np.diff(lam))
    idx = np.abs(lam - lam0) < window
    lam_win, flux_win = lam[idx], flux[idx]
    ivar_win = ivar[idx] if ivar is not None else np.ones_like(flux_win)

    signal = np.nanmedian(flux_win)
    noise = 1 / np.sqrt(np.nanmedian(ivar_win[np.isfinite(ivar_win) & (ivar_win > 0)]))
    snr = signal / noise

    if not np.isfinite(snr) or snr < 5:
        raise RuntimeError(f"SNR too low in fitting window: {snr:.2f} for λ0 = {lam0:.1f}")

    if len(lam_win) < 5 or np.all(np.isnan(flux_win)):
        raise RuntimeError("Insufficient data in window")

    try:
        # STEP 1: Voigt + spline
        methods.append("voigt")
        p0 = [0.1, lam0, 0.1, 0.1, 1.0]
        popt, pcov = curve_fit(_voigt_profile, lam_win, flux_win, p0=p0, sigma=1/np.sqrt(ivar_win), maxfev=10000)
        mu, mu_err = popt[1], np.sqrt(np.diag(pcov))[1]
        if mu_err > 10: raise RuntimeError("Unreasonably large error")
        return mu, mu_err,"" , "voigt"
    except Exception as e:
        rejection = f"voigt failed: {e}"

    try:
        # STEP 2: Skewed Gaussian
        methods.append("skewed_gauss")
        from scipy.special import erf
        p0 = [0.1, lam0, 0.1, 0.5, 1.0]
        popt, pcov = curve_fit(_skew_gauss, lam_win, flux_win, p0=p0, sigma=1/np.sqrt(ivar_win), maxfev=10000)
        mu, mu_err = popt[1], np.sqrt(np.diag(pcov))[1]
        if mu_err > 10: raise RuntimeError("Unreasonably large error")
        return mu, mu_err,"", "skewed_gauss"
    except Exception as e:
        rejection = f"skewed_gauss failed: {e}"

    try:
        # STEP 3: Asymmetric pseudo-Voigt
        methods.append("pseudo_voigt_asym")
        p0 = [0.1, lam0, 0.1, 0.1, 0.5, 1.0]
        popt, pcov = curve_fit(_pseudo_voigt_asym, lam_win, flux_win, p0=p0, sigma=1/np.sqrt(ivar_win), maxfev=10000)
        mu, mu_err = popt[1], np.sqrt(np.diag(pcov))[1]
        if mu_err > 10: raise RuntimeError("Unreasonably large error")
        return mu, mu_err,"", "pseudo_voigt_asym"
    except Exception as e:
        rejection = f"pseudo_voigt_asym failed: {e}"

    try:
        # STEP 4: GP model (stubbed here)
        methods.append("gp_model")
        raise RuntimeError("GP model not implemented")
    except Exception as e:
        rejection = f"gp_model failed: {e}"

    try:
        # STEP 5: Fallback spline
        methods.append("spline_min")
        mu, mu_err = _spline_min(lam_win, flux_win)
        return mu, mu_err,"Defaulted to Spline", "spline_min"
    except Exception as e:
        rejection = f"spline_min failed: {e}"

    raise RuntimeError(f"Ladder failed: {rejection}")
