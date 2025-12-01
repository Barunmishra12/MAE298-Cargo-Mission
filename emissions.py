import numpy as np
import matplotlib.pyplot as plt


def compute_emissions(prob, total_fuel_burn, make_plots=True):
    """
    Compute emissions for climb_1, cruise, and descent_1 phases
    using Aviary trajectory timeseries outputs.

    Parameters
    ----------
    prob : AviaryProblem
        A fully built and executed Aviary problem.
    make_plots : bool
        Whether to generate plots.

    Returns
    -------
    dict
        Total emissions in kilograms for HC, CO, NOx, nvPM.
    """

    phases = ['climb_1', 'cruise', 'descent_1']

    throttle_data = {}
    thrust_data = {}
    fuel_flow_data = {}
    time = []

    # Extract data from each phase
    for phase in phases:
        throttle_data[phase] = prob.get_val(f'traj.{phase}.timeseries.throttle')
        thrust_data[phase] = prob.get_val(f'traj.{phase}.timeseries.thrust_net_total')
        fuel_flow_data[phase] = np.abs(
            prob.get_val(f'traj.{phase}.timeseries.fuel_flow_rate_negative_total')
        )

        # Build cumulative time vector
        t_phase = prob.get_val(f'traj.{phase}.timeseries.time')
        if time:
            t_phase = t_phase + time[-1][-1]
        time.append(t_phase)

    # Concatenate arrays
    time_all = np.concatenate(time)
    throttle_all = np.concatenate([throttle_data[p] for p in phases])
    thrust_all = np.concatenate([thrust_data[p] for p in phases])
    fuel_flow_all = np.concatenate([fuel_flow_data[p] for p in phases])

    # =====================
    # Fuel flow conversions
    # =====================
    time_all_s = time_all.flatten()
    fuel_flow_lbm_hr = fuel_flow_all.flatten()

    lbm_to_kg = 0.45359237
    fuel_flow_kg_s = (fuel_flow_lbm_hr * lbm_to_kg) / 3600.0

    def HC(f):
        return 0.0317 * f ** (-1.008)
    
    def CO(f):
        return 0.12 * f ** (-1.96)
    def NOx(f):
        return 4.5452 * np.exp(2.3056 * f)
    def nvPM(f):
        return 62.891 * f ** 3 - 117.94 * f ** 2 + 59.352 * f - 4.143

    # Emissions (g/kg)
    HC_per_kg = HC(fuel_flow_kg_s)
    CO_per_kg = CO(fuel_flow_kg_s)
    NOx_per_kg = NOx(fuel_flow_kg_s)
    nvPM_per_kg = nvPM(fuel_flow_kg_s) * 1e-3  # mg/kg → g/kg

    # Instantaneous g/s emissions
    HC_rate = HC_per_kg * fuel_flow_kg_s
    CO_rate = CO_per_kg * fuel_flow_kg_s
    NOx_rate = NOx_per_kg * fuel_flow_kg_s
    nvPM_rate = nvPM_per_kg * fuel_flow_kg_s

    # Integrating total emissions
    HC_total = np.trapezoid(HC_rate, time_all_s) / 1000
    CO_total = np.trapezoid(CO_rate, time_all_s) / 1000
    NOx_total = np.trapezoid(NOx_rate, time_all_s) / 1000
    nvPM_total = np.trapezoid(nvPM_rate, time_all_s) / 1000

    total_fuel_kg = total_fuel_burn*0.453592
    # ===============================
    # CO₂ EMISSIONS (kg)
    # ===============================
    CO2_total = total_fuel_kg * 3.16

    results = {
        "Total Fuel": total_fuel_kg,
        "HC": HC_total,
        "CO": CO_total,
        "NOx": NOx_total,
        "nvPM": nvPM_total,
        "CO2": CO2_total,
    }

    print("\n===== TOTAL MISSION EMISSIONS =====")
    for k, v in results.items():
        print(f"{k}: {float(v):.3f} kg")
    print("===================================\n")

    if make_plots:
        # Timeseries plots
        plt.figure(figsize=(10,6))
        plt.subplot(3,1,1)
        plt.plot(time_all / 60, throttle_all, label='Throttle')
        plt.ylabel("Throttle")
        plt.grid(True)
        plt.legend()

        plt.subplot(3,1,2)
        plt.plot(time_all / 60, thrust_all, label='Thrust')
        plt.ylabel("Thrust (lbf)")
        plt.grid(True)
        plt.legend()

        plt.subplot(3,1,3)
        plt.plot(time_all / 60, fuel_flow_all, label='Fuel Flow')
        plt.ylabel("Fuel Flow (lb/hr)")
        plt.xlabel("Time (min)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Emission rates
        plt.figure(figsize=(12, 8))
        plt.plot(time_all_s, NOx_rate, label="NOx")
        plt.plot(time_all_s, CO_rate, label="CO")
        plt.xlabel("Time (s)")
        plt.ylabel("Emission Rate (g/s)")
        plt.title("NOx and CO Emissions Rates")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(time_all_s, HC_rate, label="HC")
        plt.plot(time_all_s, nvPM_rate, label="nvPM")
        plt.xlabel("Time (s)")
        plt.ylabel("Emission Rate (g/s)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results
