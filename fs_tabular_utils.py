"""
Utilities for building Aviary tabular aerodynamics from FlightStream polars.
"""
import numpy as np
import aviary.api as av


def build_tabular_from_fs(fs_npz_path, alt_target_ft=35000.0, n_CL=25):
    """Construct Aviary tabular aero from a FlightStream polar file."""

    fs_npz = np.load(fs_npz_path)

    alt_grid_m = np.asarray(fs_npz["alt_grid"]).reshape(-1)
    mach_grid = np.asarray(fs_npz["mach_grid"]).reshape(-1)
    alpha_grid = np.asarray(fs_npz["alpha_grid"]).reshape(-1)
    CL_tab = np.asarray(fs_npz["CL_tab"])
    CD_tab = np.asarray(fs_npz["CD_tab"])

    alt_grid_ft = alt_grid_m / 0.3048

    n_alt = alt_grid_ft.size
    n_mach = mach_grid.size
    n_alpha = alpha_grid.size

    expected_shape = (n_alt, n_mach, n_alpha)
    if CL_tab.shape != expected_shape or CD_tab.shape != expected_shape:
        raise RuntimeError(
            f"FS polar dimensions mismatch: CL_tab.shape={CL_tab.shape}, expected={expected_shape}"
        )

    CD0_table = np.zeros((n_alt, n_mach))
    for i in range(n_alt):
        for j in range(n_mach):
            CL_line = CL_tab[i, j, :]
            CD_line = CD_tab[i, j, :]
            if (CL_line.min() <= 0.0) and (CL_line.max() >= 0.0):
                CD0_ij = np.interp(0.0, CL_line, CD_line)
            else:
                idx_min = np.argmin(np.abs(CL_line))
                CD0_ij = CD_line[idx_min]
            CD0_table[i, j] = CD0_ij

    ALT_2D, MACH_2D = np.meshgrid(alt_grid_ft, mach_grid, indexing='ij')
    alt_samples = ALT_2D.ravel()
    mach_samples = MACH_2D.ravel()
    cd0_samples = CD0_table.ravel()

    CD0_data = av.NamedValues()
    CD0_data.set_val('altitude', alt_samples, units='ft')
    CD0_data.set_val('mach', mach_samples, units='unitless')
    CD0_data.set_val('zero_lift_drag_coefficient', cd0_samples, units='unitless')

    i_ref = int(np.argmin(np.abs(alt_grid_ft - alt_target_ft)))
    CL_min = max([CL_tab[i_ref, j, :].min() for j in range(n_mach)])
    CL_max = min([CL_tab[i_ref, j, :].max() for j in range(n_mach)])
    CL_grid = np.linspace(CL_min, CL_max, n_CL)

    CDI_table = np.zeros((n_mach, n_CL))
    for j in range(n_mach):
        CL_line = CL_tab[i_ref, j, :]
        CD_line = CD_tab[i_ref, j, :]
        if (CL_line.min() <= 0.0) and (CL_line.max() >= 0.0):
            CD0_ref = np.interp(0.0, CL_line, CD_line)
        else:
            idx_min = np.argmin(np.abs(CL_line))
            CD0_ref = CD_line[idx_min]
        CDi_line = CD_line - CD0_ref
        CDI_table[j, :] = np.interp(CL_grid, CL_line, CDi_line)

    MACH_2D, CL_2D = np.meshgrid(mach_grid, CL_grid, indexing='ij')
    mach_cdi_samples = MACH_2D.ravel()
    CL_samples = CL_2D.ravel()
    cdi_samples = CDI_table.ravel()

    CDI_data = av.NamedValues()
    CDI_data.set_val('mach', mach_cdi_samples, units='unitless')
    CDI_data.set_val('lift_coefficient', CL_samples, units='unitless')
    CDI_data.set_val('lift_dependent_drag_coefficient', cdi_samples, units='unitless')

    return CD0_data, CDI_data, cd0_samples, cdi_samples, mach_cdi_samples, CL_samples


class ARAdjustedProblem:
    """Helper that updates CDI table based on current aspect ratio and sweep."""

    def __init__(self, prob, cdi_base, CL_samples, baseline_AR, baseline_sweep_deg, e):
        self.prob = prob
        self.cdi_base = cdi_base.copy()
        self.CL_samples = CL_samples
        self.baseline_AR = baseline_AR
        self.baseline_sweep_rad = np.deg2rad(baseline_sweep_deg)
        self.e = e
        self.last_AR = baseline_AR
        self.last_sweep_rad = self.baseline_sweep_rad

    def update_cdi_for_current_AR(self):
        try:
            current_AR = self.prob.get_val(av.Aircraft.Wing.ASPECT_RATIO)[0]
            current_sweep_rad = np.deg2rad(self.prob.get_val(av.Aircraft.Wing.SWEEP)[0])

            if (abs(current_AR - self.last_AR) > 1.0e-5) or (abs(current_sweep_rad - self.last_sweep_rad) > 1.0e-6):
                ar_eff_baseline = self.baseline_AR * (np.cos(self.baseline_sweep_rad) ** 2)
                ar_eff_current = current_AR * (np.cos(current_sweep_rad) ** 2)

                k_baseline = 1.0 / (np.pi * self.e * ar_eff_baseline)
                k_current = 1.0 / (np.pi * self.e * ar_eff_current)
                delta_k = k_current - k_baseline

                induced_correction = (self.CL_samples ** 2) * delta_k
                corrected_cdi = self.cdi_base + induced_correction

                self.prob.aviary_inputs.set_val(
                    av.Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR,
                    corrected_cdi,
                    units='unitless',
                )

                self.last_AR = current_AR
                self.last_sweep_rad = current_sweep_rad
                print(f"  [AR/Sweep Correction] AR={current_AR:.2f}, sweep={np.rad2deg(current_sweep_rad):.2f} deg, Delta_k={delta_k:.6f}")
        except Exception:
            pass


def attach_ar_update_hooks(prob, ar_adjuster):
    """Patch prob.run_model and driver._run_model to refresh CDI each eval."""

    original_run_model = prob.run_model

    def run_model_with_ar_update(*args, **kwargs):
        ar_adjuster.update_cdi_for_current_AR()
        return original_run_model(*args, **kwargs)

    prob.run_model = run_model_with_ar_update

    if hasattr(prob, "driver") and hasattr(prob.driver, "_run_model"):
        _orig_driver_run_model = prob.driver._run_model

        def _driver_run_model_with_ar_update():
            ar_adjuster.update_cdi_for_current_AR()
            return _orig_driver_run_model()

        prob.driver._run_model = _driver_run_model_with_ar_update

    return prob


def run_ar_sensitivity_test(prob, CL_samples, cdi_samples, ar_values=None, e=0.85):
    """Run a quick AR-induced drag sensitivity sweep and restore CDI afterwards."""

    if ar_values is None:
        ar_values = [30.0, 35.0]

    print('\n--- AR induced-drag correction sensitivity test ---')
    ar_results = {}
    for ar in ar_values:
        try:
            prob.set_val(av.Aircraft.Wing.ASPECT_RATIO, ar)
        except Exception:
            try:
                prob.set_val('ASPECT_RATIO', ar)
            except Exception:
                pass

        k = 1.0 / (np.pi * e * ar)
        induced = (CL_samples ** 2) * k
        corrected_cdi = cdi_samples + induced

        prob.aviary_inputs.set_val(
            av.Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR,
            corrected_cdi,
            units='unitless',
        )

        try:
            prob.run_model()
            fuel = prob.get_val(av.Mission.Summary.FUEL_BURNED)
            ar_results[ar] = fuel
            print(f'AR={ar} -> fuel: {fuel}')
        except Exception as exc:
            print(f'AR test failed for AR={ar}: {exc}')

    prob.aviary_inputs.set_val(
        av.Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR,
        cdi_samples,
        units='unitless',
    )

    print('AR test summary:', ar_results)
    return ar_results


def apply_default_cost_inputs(prob, Aircraft):
    """Apply default economic inputs for the cost model."""

    prob.set_val(Aircraft.Cost.LABOR_RATE_MFG, 100.0)
    prob.set_val(Aircraft.Cost.LABOR_FACTOR, 1.0)
    prob.set_val(Aircraft.Cost.LABOR_PROD_FACTOR, 1.0)
    prob.set_val(Aircraft.Cost.FUEL_PRICE_PER_GAL, 6.5)
    prob.set_val(Aircraft.Cost.UTILIZATION_ANNUAL, 8000.0)
    prob.set_val(Aircraft.Cost.RESIDUAL_VALUE_FRACTION, 0.15)
    prob.set_val(Aircraft.Cost.DEPRECIATION_YEARS, 20.0)
    prob.set_val(Aircraft.Cost.CREW_COST_BASE, 1_000_000.0)
    prob.set_val(Aircraft.Cost.CREW_OVERHEAD_FRACTION, 0.035)
    prob.set_val(Aircraft.Cost.STORAGE_PER_MONTH, 25_000.0)
    prob.set_val(Aircraft.Cost.LIABILITY_INSURANCE, 200_000.0)
    prob.set_val(Aircraft.Cost.HULL_INSURANCE_RATE, 0.01)
    prob.set_val(Aircraft.Cost.OVERHAUL_RATE, 1.0e-4)
    prob.set_val(Aircraft.Cost.OVERHAUL_INTERVAL_HR, 20_000.0)
    return prob
