import os
import math
import numpy as np
import numpy as _np  # for robust scalar extraction
import matplotlib.pyplot as plt
import aviary.api as av

# ------------------------------------------------
# USER SETTINGS
# ------------------------------------------------

# Change this to your own aircraft deck if needed:
AIRCRAFT_CSV = 'models/aircraft/test_aircraft/advanced_single_aisle_baseline_FLOPS.csv'

TARGET_RANGE_NMI = 1600.0
PAYLOAD_LB = 52000.0

# Taxi/Reserve accounting
TAXI_MIN = 15.0
TAXI_FUEL_RATE_LB_PER_MIN = 20.0   # placeholder for ground fuel burn
RESERVE_FRAC = 0.05                # 5% of trip fuel

# Optimizer (IPOPT requires pyOptSparse)
OPTIMIZER = 'IPOPT'  # 'IPOPT' or 'SLSQP'

# Plot output directory
PLOT_DIR = 'plots'

# Phases in mission order
PHASES_IN_ORDER = ['climb_1', 'cruise', 'descent_1', 'hold']


# ------------------------------------------------
# PHASE DEFINITIONS
# ------------------------------------------------

def _base_phase_info():
    """
    Common baseline phase info (climb → cruise → descent → hold → post_mission).
    """
    phase_info = {
        'pre_mission': {
            'include_takeoff': False,
            'optimize_mass': True,
        },

        'climb_1': {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'num_segments': 2,
                'order': 3,

                'mach_optimize': False,
                'mach_polynomial_order': 1,
                'mach_initial': (0.2, 'unitless'),
                'mach_final': (0.72, 'unitless'),
                'mach_bounds': ((0.18, 0.74), 'unitless'),

                'altitude_optimize': False,
                'altitude_polynomial_order': 1,
                'altitude_initial': (0.0, 'ft'),
                'altitude_final': (30500.0, 'ft'),
                'altitude_bounds': ((0.0, 31000.0), 'ft'),

                'throttle_enforcement': 'path_constraint',

                'time_initial_bounds': ((0.0, 0.0), 'min'),
                'time_duration_bounds': ((27.0, 81.0), 'min'),
            },
            # [start_time, duration] in minutes
            'initial_guesses': {'time': ([0.0, 54.0], 'min')},
        },

        'cruise': {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'num_segments': 2,
                'order': 3,

                'mach_optimize': False,
                'mach_polynomial_order': 1,
                'mach_initial': (0.72, 'unitless'),
                'mach_final': (0.72, 'unitless'),
                'mach_bounds': ((0.70, 0.80), 'unitless'),

                'altitude_optimize': False,
                'altitude_polynomial_order': 1,
                'altitude_initial': (30500.0, 'ft'),
                'altitude_final': (31000.0, 'ft'),
                'altitude_bounds': ((30000.0, 31500.0), 'ft'),

                'throttle_enforcement': 'boundary_constraint',

                'time_initial_bounds': ((27.0, 81.0), 'min'),
                'time_duration_bounds': ((85.5, 256.5), 'min'),
            },
            # cruise: start at 54 min, end ~171 min → duration 117
            'initial_guesses': {'time': ([54.0, 117.0], 'min')},
        },

        'descent_1': {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'num_segments': 2,
                'order': 3,

                'mach_optimize': False,
                'mach_initial': (0.72, 'unitless'),
                'mach_final': (0.20, 'unitless'),
                'mach_bounds': ((0.18, 0.80), 'unitless'),

                'altitude_optimize': False,
                'altitude_initial': (31000.0, 'ft'),
                'altitude_final': (1500.0, 'ft'),
                'altitude_bounds': ((0.0, 31500.0), 'ft'),

                'throttle_enforcement': 'path_constraint',

                'time_initial_bounds': ((112.5, 337.5), 'min'),
                'time_duration_bounds': ((26.5, 79.5), 'min'),
            },
            # descent start near 171 min, duration ~54
            'initial_guesses': {'time': ([171.0, 54.0], 'min')},
        },

        'hold': {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'num_segments': 3,
                'order': 2,

                'mach_polynomial_order': 1,
                'mach_bounds': ((0.15, 0.25), 'unitless'),

                'altitude_polynomial_order': 1,
                'altitude_bounds': ((1000.0, 3000.0), 'ft'),
                'altitude_initial': (1500.0, 'ft'),
                'altitude_final': (1500.0, 'ft'),

                'throttle_enforcement': 'path_constraint',

                'time_initial_bounds': ((0.0, 700.0), 'min'),
                'time_duration_bounds': ((45.0, 45.0), 'min'),
            },
            # hold starts near 225 min, lasts 45 min
            'initial_guesses': {'time': ([225.0, 45.0], 'min')},
        },

        'post_mission': {
            'include_landing': False,
            'constrain_range': True,
            'target_range': (TARGET_RANGE_NMI, 'nmi'),
        },
    }
    return phase_info


def make_baseline_phase_info():
    """
    Baseline: fixed Mach & altitude schedules (no profile optimization).
    """
    p = _base_phase_info()

    # Explicitly ensure no optimization flags are on
    p['climb_1']['user_options'].update({
        'mach_optimize': False,
        'altitude_optimize': False,
    })

    p['cruise']['user_options'].update({
        'mach_optimize': False,
        'altitude_optimize': False,
    })

    p['descent_1']['user_options'].update({
        'mach_optimize': False,
        'altitude_optimize': False,
    })

    p['hold']['user_options'].update({
        'mach_optimize': False,
        'altitude_optimize': False,
    })

    return p


def make_optimized_phase_info():
    """
    Optimized: same structure, but cruise Mach and altitude become design variables.
    """
    p = _base_phase_info()

    # ---- Cruise: optimize Mach & Altitude ----
    p['cruise']['user_options'].update({
        'mach_optimize': True,
        'mach_bounds': ((0.70, 0.80), 'unitless'),

        'altitude_optimize': True,
        'altitude_polynomial_order': 3,
        'altitude_bounds': ((30000.0, 41000.0), 'ft'),

        'order': 3,
        'throttle_enforcement': 'boundary_constraint',
        'time_duration_bounds': ((60.0, 350.0), 'min'),
    })

    # Allow optimization in climb/descent if desired
    p['climb_1']['user_options'].update({
        'mach_optimize': True,
        'altitude_optimize': True,
    })

    p['descent_1']['user_options'].update({
        'mach_optimize': True,
        'altitude_optimize': True,
    })

    p['hold']['user_options'].update({
        'mach_optimize': False,
        'altitude_optimize': False,
    })

    return p


def _safe_optimized_phase_info():
    """
    Fallback where only cruise Mach is optimized (altitude fixed).
    Used if the first optimized run fails.
    """
    p = make_optimized_phase_info()
    p['cruise']['user_options'].update({
        'mach_bounds': ((0.71, 0.79), 'unitless'),
        'altitude_optimize': False,
        'altitude_polynomial_order': 1,
    })
    return p


# ------------------------------------------------
# RESULT READING HELPERS
# ------------------------------------------------

def _read_scalar(prob, candidates, units=None):
    """
    Try several variable names; return the first that exists as a Python float.
    """
    last_err = None
    for name in candidates:
        try:
            val = prob.get_val(name, units=units)
            arr = _np.asarray(val).reshape(-1)
            return float(arr[0])
        except Exception as e:
            last_err = e
    raise KeyError(f"None of these variables were found: {candidates}") from last_err


def _did_fail(range_nmi):
    """Detect an obviously failed run."""
    try:
        r = float(range_nmi)
    except Exception:
        return True
    return (np.isnan(r) or r <= 1.0)


# ------------------------------------------------
# MISSION PROFILE CHARTING
# ------------------------------------------------

def _read_series(prob, names_with_units):
    """
    Try multiple (name, units) candidates; return (array, units) or (None, None).
    """
    for n, u in names_with_units:
        try:
            val = prob.get_val(n, units=u)
            return np.atleast_1d(val), u
        except Exception:
            continue
    return None, None


def _concat_histories(prob, phases, varname, units=None, alt_names=()):
    """
    Build mission-wide history for a Dymos timeseries variable.
    Returns (time_min, values, available_bool).
    """
    t_all, y_all = [], []
    t_offset = 0.0
    base_names = (varname,) + tuple(alt_names)

    for ph in phases:
        # time
        t, u_t = _read_series(prob, [
            (f'{ph}.timeseries.time', 'min'),
            (f'{ph}.timeseries.time', 's'),
            (f'traj.phases.{ph}.timeseries.time', 'min'),
            (f'traj.phases.{ph}.timeseries.time', 's'),
        ])
        if t is None:
            continue

        # convert to minutes if seconds
        if u_t == 's':
            t = t / 60.0

        # variable
        y = None
        for base in base_names:
            y, _u = _read_series(prob, [
                (f'{ph}.timeseries.{base}', units or None),
                (f'traj.phases.{ph}.timeseries.{base}', units or None),
            ])
            if y is not None:
                break

        if y is None:
            t_offset += float(t[-1] - t[0])
            continue

        # shift time so profiles are continuous
        t_shift = t - t[0] + t_offset
        t_offset = float(t_shift[-1])

        t_all.append(np.asarray(t_shift))
        y_all.append(np.asarray(y))

    if not t_all:
        return None, None, False

    return np.concatenate(t_all), np.concatenate(y_all), True


def plot_mission_profiles(prob, tag='baseline', outdir=PLOT_DIR):
    """
    Save altitude, Mach, throttle (if available), and cumulative fuel charts.
    """
    os.makedirs(outdir, exist_ok=True)
    phases = PHASES_IN_ORDER

    # ALTITUDE
    t_alt, alt, ok_alt = _concat_histories(prob, phases, 'altitude', units='ft', alt_names=('h',))
    if ok_alt:
        plt.figure()
        plt.plot(t_alt, alt)
        plt.xlabel('Mission time [min]')
        plt.ylabel('Altitude [ft]')
        plt.title(f'Altitude Profile ({tag})')
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'altitude_vs_time_{tag}.png'), dpi=160)
        plt.close()

    # MACH
    t_m, M, ok_m = _concat_histories(prob, phases, 'mach', units=None, alt_names=('Mach',))
    if ok_m:
        plt.figure()
        plt.plot(t_m, M)
        plt.xlabel('Mission time [min]')
        plt.ylabel('Mach [-]')
        plt.title(f'Mach Profile ({tag})')
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'mach_vs_time_{tag}.png'), dpi=160)
        plt.close()

    # THROTTLE
    t_th, th, ok_th = _concat_histories(
        prob, phases, 'throttle', units=None,
        alt_names=('throttle_set', 'controls:throttle')
    )
    if ok_th:
        plt.figure()
        plt.plot(t_th, th)
        plt.xlabel('Mission time [min]')
        plt.ylabel('Throttle [-]')
        plt.title(f'Throttle Profile ({tag})')
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'throttle_vs_time_{tag}.png'), dpi=160)
        plt.close()

    # FUEL cumulative
    t_fb, fb, ok_fb = _concat_histories(
        prob, phases, 'fuel_burn', units='lb',
        alt_names=('propulsion.fuel_burn', 'fuelburn')
    )
    if ok_fb:
        fuel_cum = np.maximum.accumulate(fb) if np.any(np.diff(fb) < 0) else fb
        t_f = t_fb
    else:
        t_mas, m, ok_mas = _concat_histories(
            prob, phases, 'mass', units='lb',
            alt_names=('total_mass',)
        )
        if ok_mas:
            m0 = float(m[0])
            fuel_cum = m0 - m
            t_f = t_mas
        else:
            fuel_cum, t_f = None, None

    if fuel_cum is not None:
        plt.figure()
        plt.plot(t_f, fuel_cum)
        plt.xlabel('Mission time [min]')
        plt.ylabel('Cumulative fuel [lb]')
        plt.title(f'Fuel Use ({tag})')
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fuel_vs_time_{tag}.png'), dpi=160)
        plt.close()


def compare_mission_profiles(prob_base, prob_opt, outdir=PLOT_DIR):
    """
    Overlay baseline and optimized profiles for altitude, Mach, throttle, and fuel.
    """
    os.makedirs(outdir, exist_ok=True)
    phases_base = PHASES_IN_ORDER
    phases_opt = PHASES_IN_ORDER

    # ALTITUDE
    t_b, alt_b, ok_b = _concat_histories(prob_base, phases_base, 'altitude',
                                         units='ft', alt_names=('h',))
    t_o, alt_o, ok_o = _concat_histories(prob_opt, phases_opt, 'altitude',
                                         units='ft', alt_names=('h',))
    if ok_b and ok_o:
        plt.figure()
        plt.plot(t_b, alt_b, label='Baseline')
        plt.plot(t_o, alt_o, label='Optimized', linestyle='--')
        plt.xlabel('Mission time [min]')
        plt.ylabel('Altitude [ft]')
        plt.title('Altitude Profile: Baseline vs Optimized')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'altitude_compare.png'), dpi=160)
        plt.close()

    # MACH
    t_b, M_b, ok_b = _concat_histories(prob_base, phases_base, 'mach',
                                       units=None, alt_names=('Mach',))
    t_o, M_o, ok_o = _concat_histories(prob_opt, phases_opt, 'mach',
                                       units=None, alt_names=('Mach',))
    if ok_b and ok_o:
        plt.figure()
        plt.plot(t_b, M_b, label='Baseline')
        plt.plot(t_o, M_o, label='Optimized', linestyle='--')
        plt.xlabel('Mission time [min]')
        plt.ylabel('Mach [-]')
        plt.title('Mach Profile: Baseline vs Optimized')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'mach_compare.png'), dpi=160)
        plt.close()

    # THROTTLE
    t_b, th_b, ok_b = _concat_histories(
        prob_base, phases_base, 'throttle', units=None,
        alt_names=('throttle_set', 'controls:throttle')
    )
    t_o, th_o, ok_o = _concat_histories(
        prob_opt, phases_opt, 'throttle', units=None,
        alt_names=('throttle_set', 'controls:throttle')
    )
    if ok_b and ok_o:
        plt.figure()
        plt.plot(t_b, th_b, label='Baseline')
        plt.plot(t_o, th_o, label='Optimized', linestyle='--')
        plt.xlabel('Mission time [min]')
        plt.ylabel('Throttle [-]')
        plt.title('Throttle Profile: Baseline vs Optimized')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'throttle_compare.png'), dpi=160)
        plt.close()

    # FUEL cumulative
    t_b, fb_b, ok_b = _concat_histories(
        prob_base, phases_base, 'fuel_burn', units='lb',
        alt_names=('propulsion.fuel_burn', 'fuelburn')
    )
    if ok_b:
        fuel_b = np.maximum.accumulate(fb_b) if np.any(np.diff(fb_b) < 0) else fb_b
        t_f_b = t_b
    else:
        t_b, m_b, ok_bm = _concat_histories(
            prob_base, phases_base, 'mass', units='lb',
            alt_names=('total_mass',)
        )
        if ok_bm:
            m0_b = float(m_b[0])
            fuel_b = m0_b - m_b
            t_f_b = t_b
        else:
            fuel_b, t_f_b = None, None

    t_o, fb_o, ok_o = _concat_histories(
        prob_opt, phases_opt, 'fuel_burn', units='lb',
        alt_names=('propulsion.fuel_burn', 'fuelburn')
    )
    if ok_o:
        fuel_o = np.maximum.accumulate(fb_o) if np.any(np.diff(fb_o) < 0) else fb_o
        t_f_o = t_o
    else:
        t_o, m_o, ok_om = _concat_histories(
            prob_opt, phases_opt, 'mass', units='lb',
            alt_names=('total_mass',)
        )
        if ok_om:
            m0_o = float(m_o[0])
            fuel_o = m0_o - m_o
            t_f_o = t_o
        else:
            fuel_o, t_f_o = None, None

    if fuel_b is not None and fuel_o is not None:
        plt.figure()
        plt.plot(t_f_b, fuel_b, label='Baseline')
        plt.plot(t_f_o, fuel_o, label='Optimized', linestyle='--')
        plt.xlabel('Mission time [min]')
        plt.ylabel('Cumulative fuel [lb]')
        plt.title('Fuel Use: Baseline vs Optimized')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fuel_compare.png'), dpi=160)
        plt.close()


def plot_summary_comparison(BASE, OPT, outdir=PLOT_DIR):
    """
    Bar plot comparing baseline and optimized fuel-per-ton-mile.
    """
    os.makedirs(outdir, exist_ok=True)

    labels = ['Baseline', 'Optimized']
    fpm_vals = [BASE['fuel_per_ton_mile_lb'], OPT['fuel_per_ton_mile_lb']]

    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x, fpm_vals)
    plt.xticks(x, labels)
    plt.ylabel('Fuel per ton-mile [lb/ton-mi]')
    plt.title('Fuel per Ton-Mile: Baseline vs Optimized')
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)

    for i, v in enumerate(fpm_vals):
        if np.isfinite(v) and v > 0:
            plt.text(i, v * 1.01, f"{v:.3f}",
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fuel_per_tonmile_compare.png'), dpi=160)
    plt.close()


# ------------------------------------------------
# RUN & SCORE
# ------------------------------------------------

def run_and_score(phase_info, optimizer=OPTIMIZER, tag='run'):
    """
    Run Aviary and return dict with fuel/ton-mi and details.
    Auto-retries once with a safer optimized setup if the run fails.
    """
    prob = av.run_aviary(AIRCRAFT_CSV, phase_info, optimizer=optimizer, make_plots=True)

    try:
        trip_fuel_lb = _read_scalar(
            prob,
            ['mission:summary:fuel_burned',
             'mission:objectives:fuel',
             'mission:design:fuel_mass'],
            units='lb'
        )
        range_nmi = _read_scalar(
            prob,
            ['mission:summary:range', 'mission:range'],
            units='nmi'
        )
    except KeyError:
        trip_fuel_lb, range_nmi = float('nan'), 0.0

    if tag == 'optimized' and _did_fail(range_nmi):
        print(f"[{tag}] First attempt failed (range={range_nmi}). Retrying with safer optimized settings...")
        safe_phase = _safe_optimized_phase_info()
        prob = av.run_aviary(AIRCRAFT_CSV, safe_phase, optimizer=optimizer, make_plots=True)
        trip_fuel_lb = _read_scalar(
            prob,
            ['mission:summary:fuel_burned',
             'mission:objectives:fuel',
             'mission:design:fuel_mass'],
            units='lb'
        )
        range_nmi = _read_scalar(
            prob,
            ['mission:summary:range', 'mission:range'],
            units='nmi'
        )

    taxi_fuel = TAXI_MIN * TAXI_FUEL_RATE_LB_PER_MIN
    try:
        reserve_fuel = _read_scalar(prob, ['mission:design:reserve_fuel'], units='lb')
    except KeyError:
        reserve_fuel = RESERVE_FRAC * trip_fuel_lb

    total_block_fuel = trip_fuel_lb + reserve_fuel + taxi_fuel

    ton_miles = (PAYLOAD_LB / 2000.0) * (1.15078 * range_nmi)
    fpm = total_block_fuel / ton_miles if ton_miles > 0 else float('inf')

    return {
        'prob': prob,
        'trip_fuel_lb': trip_fuel_lb,
        'range_nmi': range_nmi,
        'reserve_fuel_lb': reserve_fuel,
        'taxi_fuel_lb': taxi_fuel,
        'total_block_fuel_lb': total_block_fuel,
        'fuel_per_ton_mile_lb': fpm,
    }


# ------------------------------------------------
# MISSION NUMERIC SUMMARY & INTERPRETATION
# ------------------------------------------------

def _summarize_phase(prob, phase_name):
    """
    Extract metrics for one phase, supporting both 'phase' and 'traj.phases.phase'.
    """
    candidate_roots = [
        phase_name,
        f"traj.phases.{phase_name}",
    ]

    base = None
    t = None

    for root in candidate_roots:
        try:
            t = prob.get_val(f'{root}.timeseries.time', units='min').ravel()
            if t.size > 0:
                base = root
                break
        except Exception:
            continue

    if base is None or t is None or t.size == 0:
        return None

    t_start = float(t[0])
    t_end = float(t[-1])
    dt = t_end - t_start

    # Mach
    M = None
    try:
        M = prob.get_val(f'{base}.timeseries.mach').ravel()
    except Exception:
        try:
            M = prob.get_val(f'{base}.timeseries.Mach').ravel()
        except Exception:
            M = None
    M_avg = float(np.mean(M)) if M is not None and M.size > 0 else np.nan

    # Altitude
    h = None
    try:
        h = prob.get_val(f'{base}.timeseries.altitude', units='ft').ravel()
    except Exception:
        try:
            h = prob.get_val(f'{base}.timeseries.h', units='ft').ravel()
        except Exception:
            h = None
    h_avg = float(np.mean(h)) if h is not None and h.size > 0 else np.nan

    # Fuel
    fuel_phase = np.nan
    try:
        fb = prob.get_val(f'{base}.timeseries.fuel_burn', units='lb').ravel()
        if fb.size > 0:
            fuel_phase = float(fb[-1] - fb[0])
    except Exception:
        m = None
        try:
            m = prob.get_val(f'{base}.timeseries.mass', units='lb').ravel()
        except Exception:
            try:
                m = prob.get_val(f'{base}.timeseries.total_mass', units='lb').ravel()
            except Exception:
                m = None
        if m is not None and m.size > 0:
            fuel_phase = float(m[0] - m[-1])

    return {
        't_start_min': t_start,
        't_end_min': t_end,
        'dt_min': dt,
        'mach_avg': M_avg,
        'alt_avg_ft': h_avg,
        'fuel_lb': fuel_phase,
    }


def print_mission_changes(prob_base, prob_opt):
    """
    Compare baseline vs optimized mission phase-by-phase.
    """
    print("\n" + "=" * 20 + " MISSION PROFILE CHANGES " + "=" * 20)

    header = (
        f"{'Phase':<10}"
        f"{'Metric':<18}"
        f"{'Baseline':>14}"
        f"{'Optimized':>14}"
        f"{'Δ (Opt - Base)':>16}"
    )
    print(header)
    print("-" * len(header))

    for ph in PHASES_IN_ORDER:
        mb = _summarize_phase(prob_base, ph)
        mo = _summarize_phase(prob_opt, ph)

        if mb is None or mo is None:
            continue

        def row(metric_name, key, fmt="{:.3f}"):
            vb = mb[key]
            vo = mo[key]
            if np.isnan(vb) and np.isnan(vo):
                return
            d = vo - vb if (np.isfinite(vb) and np.isfinite(vo)) else np.nan
            s_b = fmt.format(vb) if np.isfinite(vb) else "nan"
            s_o = fmt.format(vo) if np.isfinite(vo) else "nan"
            s_d = fmt.format(d) if np.isfinite(d) else "nan"
            print(f"{ph:<10}{metric_name:<18}{s_b:>14}{s_o:>14}{s_d:>16}")

        row("t_start [min]", 't_start_min', "{:.2f}")
        row("t_end [min]", 't_end_min', "{:.2f}")
        row("duration [min]", 'dt_min', "{:.2f}")
        row("Mach avg [-]", 'mach_avg', "{:.4f}")
        row("Alt avg [ft]", 'alt_avg_ft', "{:.1f}")
        row("Fuel [lb]", 'fuel_lb', "{:.2f}")

        print("-" * len(header))


def interpret_mission_changes(prob_base, prob_opt):
    """
    Human-readable interpretation of mission profile changes.
    """
    print("\n" + "=" * 20 + " INTERPRETATION OF CHANGES " + "=" * 20)

    for ph in PHASES_IN_ORDER:
        mb = _summarize_phase(prob_base, ph)
        mo = _summarize_phase(prob_opt, ph)
        if mb is None or mo is None:
            continue

        print(f"\n>>> {ph.upper()} PHASE")

        d_dt = mo['dt_min'] - mb['dt_min']
        if np.isfinite(d_dt) and abs(d_dt) > 0.1:
            if d_dt > 0:
                print(f"- Duration increased by {d_dt:.1f} min (optimizer spends longer here).")
            else:
                print(f"- Duration decreased by {-d_dt:.1f} min (optimizer shortened this phase).")

        d_m = mo['mach_avg'] - mb['mach_avg']
        if np.isfinite(d_m) and abs(d_m) > 0.003:
            arrow = "↑" if d_m > 0 else "↓"
            print(
                f"- Average Mach changed {arrow} from {mb['mach_avg']:.3f} "
                f"to {mo['mach_avg']:.3f} ({d_m:+.3f})."
            )

        d_h = mo['alt_avg_ft'] - mb['alt_avg_ft']
        if np.isfinite(d_h) and abs(d_h) > 20:
            arrow = "↑" if d_h > 0 else "↓"
            print(
                f"- Average altitude changed {arrow} from {mb['alt_avg_ft']:.0f} ft "
                f"to {mo['alt_avg_ft']:.0f} ft ({d_h:+.0f} ft)."
            )

        d_f = mo['fuel_lb'] - mb['fuel_lb']
        if np.isfinite(d_f) and abs(d_f) > 5:
            arrow = "↑" if d_f > 0 else "↓"
            print(
                f"- Fuel change {arrow}: baseline {mb['fuel_lb']:.1f} lb → "
                f"optimized {mo['fuel_lb']:.1f} lb ({d_f:+.1f} lb)."
            )

        if (abs(d_dt) < 0.1 and abs(d_m) < 0.003 and abs(d_h) < 20 and abs(d_f) < 5):
            print("- No major change in this phase.")


# ------------------------------------------------
# PRINT PHASE INFO
# ------------------------------------------------

def print_phase_info(phase_info, label="PHASE INFO"):
    print("\n" + "=" * 20 + f" {label} " + "=" * 20)
    for ph_name, ph_data in phase_info.items():
        print(f"\n--- Phase: {ph_name} ---")

        if ph_name in ("pre_mission", "post_mission"):
            for k, v in ph_data.items():
                print(f"  {k}: {v}")
            continue

        uo = ph_data.get("user_options", {})
        ig = ph_data.get("initial_guesses", {})

        t_bounds = uo.get("time_initial_bounds", None)
        dt_bounds = uo.get("time_duration_bounds", None)
        t_guess = ig.get("time", None)

        print(f"  time_initial_bounds: {t_bounds}")
        print(f"  time_duration_bounds: {dt_bounds}")
        print(f"  initial_guesses[time]: {t_guess}")

        print(f"  mach_optimize: {uo.get('mach_optimize', None)}")
        print(f"  mach_initial:  {uo.get('mach_initial', None)}")
        print(f"  mach_final:    {uo.get('mach_final', None)}")
        print(f"  mach_bounds:   {uo.get('mach_bounds', None)}")

        print(f"  altitude_optimize: {uo.get('altitude_optimize', None)}")
        print(f"  altitude_initial:  {uo.get('altitude_initial', None)}")
        print(f"  altitude_final:    {uo.get('altitude_final', None)}")
        print(f"  altitude_bounds:   {uo.get('altitude_bounds', None)}")

        print(f"  throttle_enforcement: {uo.get('throttle_enforcement', None)}")


# ------------------------------------------------
# MAIN
# ------------------------------------------------

if __name__ == '__main__':

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Build phase info dicts
    baseline_phase_info = make_baseline_phase_info()
    optimized_phase_info = make_optimized_phase_info()

    # Print mission setup to help debugging
    print_phase_info(baseline_phase_info, label="BASELINE MISSION")
    print_phase_info(optimized_phase_info, label="OPTIMIZED MISSION")

    # Run baseline
    BASE = run_and_score(baseline_phase_info, optimizer=OPTIMIZER, tag='baseline')
    plot_mission_profiles(BASE['prob'], tag='baseline', outdir=PLOT_DIR)

    # Run optimized
    OPT = run_and_score(optimized_phase_info, optimizer=OPTIMIZER, tag='optimized')
    plot_mission_profiles(OPT['prob'], tag='optimized', outdir=PLOT_DIR)

    # Overlay mission profiles
    compare_mission_profiles(BASE['prob'], OPT['prob'], outdir=PLOT_DIR)

    # Summary bar plot
    plot_summary_comparison(BASE, OPT, outdir=PLOT_DIR)

    # Numeric mission-profile comparison
    print_mission_changes(BASE['prob'], OPT['prob'])

    # Human-readable interpretation
    interpret_mission_changes(BASE['prob'], OPT['prob'])

    # Print results summary
    print("\nOptimization Complete")
    print("-----------------------------------")

    print("\n=== BASELINE (Same Aircraft, with Hold) ===")
    print(f"Trip fuel (no taxi/reserve) [lb]: {BASE['trip_fuel_lb']:.4f}")
    print(f"Range [nmi]:                      {BASE['range_nmi']:.1f}")
    print(f"Total block fuel [lb]:            {BASE['total_block_fuel_lb']:.4f}")
    print(f"Payload [lb]:                     {PAYLOAD_LB:.1f}")
    print(f"Fuel per ton-mile [lb/ton-mi]:    {BASE['fuel_per_ton_mile_lb']:.5f}")

    print("\n=== OPTIMIZED (Same Aircraft, with Hold) ===")
    print(f"Trip fuel (no taxi/reserve) [lb]: {OPT['trip_fuel_lb']:.4f}")
    print(f"Range [nmi]:                      {OPT['range_nmi']:.1f}")
    print(f"Total block fuel [lb]:            {OPT['total_block_fuel_lb']:.4f}")
    print(f"Payload [lb]:                     {PAYLOAD_LB:.1f}")
    print(f"Fuel per ton-mile [lb/ton-mi]:    {OPT['fuel_per_ton_mile_lb']:.5f}")

    base_fpm = BASE['fuel_per_ton_mile_lb']
    opt_fpm = OPT['fuel_per_ton_mile_lb']

    print("\n--- DEBUG: raw fuel_per_ton_mi ---")
    print(f"Baseline fuel_per_ton_mi = {base_fpm}")
    print(f"Optimized fuel_per_ton_mi = {opt_fpm}")
    print("-------------------------------")

    valid_baseline = np.isfinite(base_fpm) and (base_fpm > 0.0)
    valid_opt = np.isfinite(opt_fpm) and (opt_fpm > 0.0) and (OPT['range_nmi'] > 1.0)

    if valid_baseline and valid_opt:
        improvement = 100.0 * (base_fpm - opt_fpm) / base_fpm
        print(f"\nFuel/ton-mi improvement: {improvement:.2f}%")
    else:
        print("\nCannot compute improvement: baseline or optimized fuel-per-ton-mile is zero or invalid.")

    print("\nMission plots saved to: ./plots/")
