import concurrent.futures as cf
import io, os, csv, requests
from pathlib import Path
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import uuid
from fit_absorption_line import fit_absorption_line_ladder as _fit_line
from astropy.constants import c

matplotlib.use("Agg")

CSV_PATH      = "MyTable_jxu.csv"
SDSS_ROOT     = ("https://data.sdss.org/sas/dr16/eboss/spectro/redux/"
                 "v5_13_0/spectra")
N_WORKERS     = 10            # saturate bandwidth, still polite
TIMEOUT       = 60            # s

REDUX_VERSIONS = ["v5_13_0", "v5_13_2", "v6_0_4"]

# Rest wavelengths (in Ångströms) and many–multiplet sensitivity coefficients (dimensionless)
# Transition table with accurate K values and source citations
TRANSITIONS = [
    # High K-value transitions first (maximizes ΔK leverage)
    ("MnII_2576", 2576.88, 0.0350),  # Berengut et al. (2004)
    ("MnII_2594", 2594.50, 0.0340),  # Berengut et al. (2004)
    ("MnII_2606", 2606.46, 0.0320),  # Berengut et al. (2004)

    ("NiII_1709", 1709.60, 0.0280),  # Dzuba et al. (1999)
    ("NiII_1741", 1741.55, 0.0270),  # Dzuba et al. (1999)
    ("NiII_1751", 1751.92, 0.0240),  # Dzuba et al. (1999)

    ("TiII_1910", 1910.60, 0.0250),  # Berengut et al. (2004)
    ("CrII_2056", 2056.26, 0.0200),  # Berengut et al. (2004)
    ("CrII_2062", 2062.23, 0.0180),  # Berengut et al. (2004)

    ("ZnII_2026", 2026.14, 0.0180),  # Berengut et al. (2004)
    ("ZnII_2062", 2062.66, 0.0160),  # Berengut et al. (2004)

    ("CIV_1548", 1548.204, 0.1908),  # Murphy et al. (2001) — [Relativistic Many-Body]
    ("CIV_1550", 1550.781, 0.0952),  # Murphy et al. (2001)

    # Lower K-values follow; useful for contrast in Δα calculations
    ("SiIV_1393", 1393.76, 0.0066),  # Murphy et al. (2001)
    ("SiIV_1402", 1402.77, 0.0037),  # Murphy et al. (2001)
    ("CII_1334", 1334.53, 0.0013),   # Dzuba et al. (1999)
    ("AlII_1670", 1670.79, 0.0014),  # Berengut et al. (2004)
    ("SiII_1260", 1260.42, 0.0015),  # Murphy et al. (2001)
    ("OI_1302", 1302.17, 0.0000),    # OI is insensitive — included for reference only
]

PLOT_COUNT = 0
DEBUG = False
WARNING = False

def plot_sample_windows(lam, flux, z, lam0, tag):
    if DEBUG:
        print(f"[DEBUG] lam = {lam}, flux = {flux}, z = {z}, lam0 = {lam0}, tag = {tag}")
    if lam is None or flux is None or len(lam) == 0 or np.all(np.isnan(flux)):
        if WARNING:
            print(f"[WARNING] Skipping plot for {tag}: empty or invalid flux")
        return

    fig = Figure(figsize=(6, 4))
    FigureCanvas(fig)  # Required to bind backend

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lam, flux, marker='o', lw=1)
    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Flux")
    ax.set_title(f"Quasar z={z:.5f}")

    path = f"{tag}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)       

def url_for(plate, mjd, fiber):
    plate4 = f"{int(plate):04d}"
    fiber4 = f"{int(fiber):04d}"
    base   = "https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0"

    paths = [
        f"{base}/spectra/full/{plate4}/spec-{plate4}-{mjd}-{fiber4}.fits",
        f"{base}/spectra/{plate4}/spec-{plate4}-{mjd}-{fiber4}.fits",
        f"{base}/spectra/lite/{plate4}/spec-{plate4}-{mjd}-{fiber4}.fits",
    ]

    for url in paths:
        try:
            if requests.head(url, timeout=10).status_code == 200:
                return url
        except requests.RequestException:
            pass            # network hiccup → keep probing

    return None              # all three failed


# ---------------- public API -----------------------------
def fit_many_multiplet(lam: np.ndarray,
                       flux: np.ndarray,
                       ivar: np.ndarray,
                       z: float) -> dict[str, float]:
    """
    Return Δα/α and 1 σ error using the many‑multiplet method.

    Parameters
    ----------
    lam  : np.ndarray   observed‑frame wavelengths (Å)
    flux : np.ndarray   normalized flux (continuum ≈ 1)
    ivar : np.ndarray   inverse variance
    z    : float        absorber redshift

    Returns
    -------
    dict with keys 'delta_alpha', 'delta_alpha_err'
    """
    cents, errs, Ks, lam_exp, offsets, fit_methods, rejection_reasons = [], [], [], [], [], [], []
    lam_min = lam.min()
    lam_max = lam.max()
    window = 1.5  # or whatever you're using
    median_spacing = np.median(np.diff(lam))
    default_window = 6 * median_spacing  # spans ~±3σ for most Gaussian profiles

    # Filter transitions to those that fall within the observable spectrum window
    valid_transitions = [
        (name, lam0_rest, K)
        for name, lam0_rest, K in TRANSITIONS
        if lam0_rest * (1 + z) >= lam_min + default_window and
           lam0_rest * (1 + z) <= lam_max - default_window
    ]

    if len(valid_transitions) < 2:
        if WARNING:
            print(f"[WARNING] Not enough transitions in observable window: "
                  f"{len(valid_transitions)} < 2")  
        return {
            "delta_alpha": None,
            "delta_alpha_err": None,
            "fit_methods": "",
            "rejection_reason": "not enough transitions in observable window",
            "mu_offsets": "",
            "mu_errors": ""
        }

    for name, lam0_rest, K in valid_transitions:
        lam0_obs = lam0_rest * (1 + z)
        if lam0_obs < lam.min() + window or lam0_obs > lam.max() - window:
            if WARNING:
                print(f"[WARNING] {name} out of range: {lam0_obs:.1f} not in [{lam_min:.1f}, {lam_max:.1f}]")
            continue  # skip out-of-range lines
        if DEBUG:
            print(f"[DEBUG] Trying to fit {name} at λ_obs = {lam0_obs:.1f} for z = {z:.3f}")
        try:
            mu, mu_err, rejection_reason, fit_method = _fit_line(lam=lam, flux=flux, ivar=ivar, lam0=lam0_obs)
            
            mu_offset = abs(mu - lam0_obs)
            
            # Reject cases based on quality control
            v_offset = abs(mu - lam0_obs) / lam0_obs * c.to("km/s").value
            if np.isnan(mu) or np.isnan(mu_err):
                rejection_reason += " nan_result"
                continue
            elif mu_err > 10:
                rejection_reason += " mu_err_too_large"
                continue
            elif v_offset > 50:
                rejection_reason += " mu_far_from_expected"
                continue
            elif np.nanmin(flux) > 0.95 * np.nanmax(flux):
                rejection_reason += " flat_flux"
                continue
            
            if DEBUG:
                print(f"[SUCCESS] Line {name} → μ = {mu:.2f} ± {mu_err:.2f} (offset = {mu_offset:.2f})")
            cents.append(mu)
            errs.append(mu_err)
            Ks.append(K)
            lam_exp.append(lam0_obs)
            offsets.append(mu_offset)
            fit_methods.append(fit_method)
            rejection_reasons.append(name + rejection_reason or "")

        except RuntimeError as exc:
            if WARNING:
                print(f"[WARNING] {name} fit failed: {exc}")
            rejection_reason = ("fit_failed " + str(exc))
        
    if len(cents) < 2:
        return {
            "delta_alpha": None,
            "delta_alpha_err": None,
            "fit_methods": ",".join(fit_methods),
            "rejection_reason": ",".join(rejection_reasons),
            "mu_offsets": ",".join(f"{x:.3f}" for x in offsets),
        }

    cents, errs, Ks, lam_exp = map(np.array, (cents, errs, Ks, lam_exp))
    dlam_lam = (cents - lam_exp) / lam_exp        # fractional shifts
    with np.errstate(divide='ignore'):
        W = np.where(errs > 0, 1 / errs**2, 0.0)

    S_k  = np.sum(W * Ks**2)
    S_yk = np.sum(W * Ks * dlam_lam)

    delta_alpha = S_yk / S_k
    if not np.isfinite(delta_alpha):
        if WARNING:
            print("[WARNING] Non-finite slope detected, clamping to 0")
        delta_alpha = 0.0

    delta_alpha_err = np.sqrt(1 / S_k)

    return {
        "delta_alpha": float(delta_alpha),
        "delta_alpha_err": float(delta_alpha_err),
        "fit_methods": ",".join(fit_methods),
        "rejection_reason": ",".join(rejection_reasons),
        "mu_offsets": ",".join(f"{x:.3f}" for x in offsets),
    }

def process_row(row):
    url = url_for(row["plate"], row["mjd"], row["fiberID"])
    if url is None:
        return {"status": "missing", **row}

    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with fits.open(io.BytesIO(r.content), memmap=False) as hdul:
                data = hdul[1].data                 # co‑added spectrum
                lam  = 10**data["loglam"]
                flux = data["flux"]
                ivar = data["ivar"]

                # ---- science callback ------------------------------------
                measurement = fit_many_multiplet(lam, flux, ivar, float(row["z"]))
                
                
                if (measurement.get("delta_alpha") is None or
                    not np.isfinite(measurement.get("delta_alpha")) or
                    measurement.get("delta_alpha_err") is None):
                    return {"status": "fail", **row, **measurement}
                
                if measurement.get("delta_alpha_err") > 10:
                    measurement.update({"rejection_reason": f"large_error {measurement['delta_alpha_err']}"})
                
                return {"status": "ok", **row, **measurement}
    except Exception as exc:
        if WARNING:
            print(f"[WARNING] {url} → {exc}")
        # guarantee the dict shape stays consistent
        measurement = {"delta_alpha": None, "delta_alpha_err": None}
        return {"status": "fail", **row, **measurement}

def run():
    rows = list(csv.DictReader(open(CSV_PATH)))
    out  = []
    with cf.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for res in pool.map(process_row, rows):
            if res: # and res.get("status") == "ok": 
                out.append(res)
    # ---------- STEP 4 GUARD  -----------------------------------------
    if not any(r.get("status", "ok") == "ok" for r in out):
        if WARNING:
            print("[WARNING] No successful spectra downloaded.")
        #return                     # or raise RuntimeError(...) to hard‑fail
    # ------------------------------------------------------------------
    
    # write aggregated results    
    keys = out[0].keys()
    with open("quasar_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(keys))
        writer.writeheader(); writer.writerows(out)

if __name__ == "__main__":
    run()

