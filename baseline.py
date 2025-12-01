import numpy as np
import openmdao.api as om
import aviary.api as av
import matplotlib.pyplot as plt

# Path to aircraft definition
aircraft_filename = 'models/aircraft/advanced_single_aisle/advanced_single_aisle_baseline_FLOPS.csv'
optimizer = 'SLSQP'
make_plots = True
max_iter = 100

phase_info = {
    'pre_mission': {'include_takeoff': False, 'optimize_mass': True},
    'climb_1': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_polynomial_order': 1,
            'mach_initial': (0.2, 'unitless'),
            'mach_final': (0.72, 'unitless'),
            'mach_bounds': ((0.18, 0.84), 'unitless'),
            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (0.0, 'ft'),
            'altitude_final': (32500.0, 'ft'),
            'altitude_bounds': ((0.0, 33000.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'time_initial_bounds': ((0.0, 0.0), 'min'),
            'time_duration_bounds': ((35.0, 105.0), 'min'),
        },
        'initial_guesses': {'time': ([0, 70], 'min')},
    },
    'cruise': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_polynomial_order': 1,
            'mach_initial': (0.72, 'unitless'),
            'mach_final': (0.80, 'unitless'),
            'mach_bounds': ((0.7, 0.84), 'unitless'),
            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (32500.0, 'ft'),
            'altitude_final': (36000.0, 'ft'),
            'altitude_bounds': ((32000.0, 36500.0), 'ft'),
            'throttle_enforcement': 'boundary_constraint',
            'time_initial_bounds': ((35.0, 105.0), 'min'),
            'time_duration_bounds': ((91.5, 274.5), 'min'),
        },
        'initial_guesses': {'time': ([70, 183], 'min')},
    },
    'descent_1': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_polynomial_order': 1,
            'mach_initial': (0.72, 'unitless'),
            'mach_final': (0.21, 'unitless'),
            'mach_bounds': ((0.19, 0.84), 'unitless'),
            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (36000.0, 'ft'),
            'altitude_final': (0.0, 'ft'),
            'altitude_bounds': ((0.0, 36500.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'time_initial_bounds': ((126.5, 379.5), 'min'),
            'time_duration_bounds': ((25.0, 75.0), 'min'),
        },
        'initial_guesses': {'time': ([253, 50], 'min')},
    },
    'post_mission': {
        'include_landing': False,
        'constrain_range': True,
        'target_range': (1600, 'nmi'),
    },
}

prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs(aircraft_filename, phase_info)

prob.check_and_preprocess_inputs()

prob.build_model()

prob.add_driver(optimizer, max_iter=max_iter)

prob.add_design_variables()
prob.add_objective()


prob.setup()
prob.run_aviary_problem(make_plots=make_plots)

# Geometry Outputs
MTOW = prob.get_val(av.Mission.Design.GROSS_MASS)[0]
AR = prob.get_val(av.Aircraft.Wing.ASPECT_RATIO)
S = prob.get_val(av.Aircraft.Wing.AREA)
taper = prob.get_val(av.Aircraft.Wing.TAPER_RATIO)
sweep = prob.get_val(av.Aircraft.Wing.SWEEP)
fuselage_length = prob.get_val(av.Aircraft.Fuselage.LENGTH)
fuselage_width = prob.get_val(av.Aircraft.Fuselage.MAX_HEIGHT)
fuselage_height = prob.get_val(av.Aircraft.Fuselage.MAX_WIDTH)

# Derived geometric quantities needed by OAS
span = np.sqrt(AR * S)              # b = sqrt(AR * S)
c_avg = S / span                    # mean chord
mach = 0.78
alt_ft = 35000.0
alt_m = alt_ft * 0.3048
T = 218.8
a = np.sqrt(1.4 * 287.05 * T)
rho = 0.38  # kg/m^3 typical at 35k ft
n_load = 2.5

oas_inputs = {
    "aspect_ratio": float(AR),
    "wing_area": float(S),
    "span": float(span),
    "mean_chord": float(c_avg),
    "taper_ratio": float(taper),
    "sweep_deg": float(sweep),
    "aircraft_mass": float(MTOW),
    "rho": rho,
    "mach": mach,
    "speed_of_sound": a,
    "load_factor": n_load,
}

outputs = {
    "aspect_ratio": float(AR),
    "wing_area [ft^2]": float(S),
    "span [ft]": float(span),
    "mean_chord [ft]": float(c_avg),
    "taper_ratio": float(taper),
    "sweep_deg [deg]": float(sweep),
    "MTOW [lbm]": float(MTOW),
    "fuselage_length [ft]": float(fuselage_length),
    "fuselage_width [ft]": float(fuselage_width),
    "fuselage_height [ft]": float(fuselage_height),
    "range [nmi]": prob.get_val(av.Mission.Summary.RANGE),
    "total_fuel_burn [lbm]": prob.get_val(av.Mission.Summary.FUEL_BURNED),
}

print("============= OUTPUTS =============")
for k, v in outputs.items():
    print(f"{k:20s} : {v}")
print("===================================")



# Emissions Estimation
from emissions import compute_emissions
emissions = compute_emissions(prob, make_plots=True)

# Structural Analysis
from structure import compute_structure
structural_results = compute_structure(oas_inputs,plot_results=False)

