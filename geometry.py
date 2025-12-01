import numpy as np
import openmdao.api as om
import aviary.api as av

# Path to your aircraft definition file
aircraft_filename = 'models/aircraft/advanced_single_aisle/advanced_single_aisle_baseline_FLOPS.csv'
optimizer = 'SLSQP'
make_plots = True
max_iter = 100

phase_info = {
    'pre_mission': {
        'include_takeoff': False,
        'optimize_mass': False,
    },

    'climb_1': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
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
        'initial_guesses': {
            'time': ([0.0, 54.0], 'min'),
        },
    },

    'cruise': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,

            'mach_optimize': False,
            'mach_polynomial_order': 1,
            'mach_initial': (0.72, 'unitless'),
            'mach_final': (0.72, 'unitless'),
            'mach_bounds': ((0.7, 0.8), 'unitless'),

            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (30500.0, 'ft'),
            'altitude_final': (31000.0, 'ft'),
            'altitude_bounds': ((30000.0, 31500.0), 'ft'),

            'throttle_enforcement': 'boundary_constraint',

            'time_initial_bounds': ((27.0, 81.0), 'min'),
            'time_duration_bounds': ((85.5, 256.5), 'min'),
        },
        'initial_guesses': {
            'time': ([54.0, 117.0], 'min'),
        },
    },

    'descent_1': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,

            'mach_optimize': False,
            'mach_polynomial_order': 1,
            'mach_initial': (0.72, 'unitless'),
            'mach_final': (0.2, 'unitless'),
            'mach_bounds': ((0.18, 0.8), 'unitless'),

            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (31000.0, 'ft'),
            'altitude_final': (1500.0, 'ft'),
            'altitude_bounds': ((0.0, 31500.0), 'ft'),

            'throttle_enforcement': 'path_constraint',

            'time_initial_bounds': ((112.5, 337.5), 'min'),
            'time_duration_bounds': ((26.5, 79.5), 'min'),
        },
        'initial_guesses': {
            'time': ([171.0, 54.0], 'min'),
        },
    },

    'hold': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 3,
            'order': 2,

            'mach_optimize': False,
            'mach_initial': None,
            'mach_final': None,
            'mach_bounds': ((0.15, 0.25), 'unitless'),

            'altitude_optimize': False,
            'altitude_initial': (1500.0, 'ft'),
            'altitude_final': (1500.0, 'ft'),
            'altitude_bounds': ((1000.0, 3000.0), 'ft'),

            'throttle_enforcement': 'path_constraint',

            'time_initial_bounds': ((0.0, 700.0), 'min'),
            'time_duration_bounds': ((45.0, 45.0), 'min'),
        },
        'initial_guesses': {
            'time': ([225.0, 45.0], 'min'),
        },
    },

    'post_mission': {
        'include_landing': False,
        'constrain_range': True,
        'target_range': (1600.0, 'nmi'),
    },
}

prob = av.AviaryProblem()
prob.load_inputs(aircraft_filename, phase_info)

prob.check_and_preprocess_inputs()

prob.build_model()

prob.add_driver(optimizer, max_iter=max_iter)

# ===================== Design Variables ==========================
prob.add_design_variables()
# Aspect Ratio
prob.model.add_design_var(av.Aircraft.Wing.ASPECT_RATIO, lower=10.0, upper=20, ref=10.0)
# Wing Area
prob.model.add_design_var(av.Aircraft.Wing.AREA, lower=500.0, upper=1600.0, ref=600.0)
# Wing Sweep
prob.model.add_design_var(av.Aircraft.Wing.SWEEP, lower=0.0, upper=30.0, ref=5.0)
# Taper Ratio
prob.model.add_design_var(av.Aircraft.Wing.TAPER_RATIO, lower=0.1, upper=0.5, ref=0.26419)
# Dihedral Angle
prob.model.add_design_var(av.Aircraft.Wing.DIHEDRAL, lower=-5.0, upper=15.0, ref=6.0)
# Fuselage Length
prob.model.add_design_var(av.Aircraft.Fuselage.LENGTH, lower=100.0, upper=130.0, ref=125.0)
# Fuselage Max Height
prob.model.add_design_var(av.Aircraft.Fuselage.MAX_HEIGHT, lower=10.0, upper=15.0, ref=13.0)
# Fuselage Max Width
prob.model.add_design_var(av.Aircraft.Fuselage.MAX_WIDTH, lower=10.0, upper=15.0, ref=12.3)
# Horziontal Tail
prob.model.add_design_var(av.Aircraft.HorizontalTail.AREA, lower=325.0, upper=380.0, ref=360.0)
prob.model.add_design_var(av.Aircraft.HorizontalTail.ASPECT_RATIO, lower=4.0, upper=6.0, ref=5.0)
prob.model.add_design_var(av.Aircraft.HorizontalTail.TAPER_RATIO, lower=0.0, upper=0.5, ref=0.27)

# ===================== CONSTRAINTS ==========================
# Boeing 737-9 takeoff wing loading
W_S_737 = 142.5  # lb/ft^2

# Tail Volume 
Vh_min = 0.2
Vh_max = 1.3


Vh_comp = om.ExecComp(
    "Vh = (S_h * l_h) / (S_w * MAC); b = (AR * S_w)**2; c_root = 2*S_w/(b*(1+taper)); MAC = (2/3)*c_root*(1+taper+taper**2)/(1+taper); l_h = 0.4*fuse_length",
    S_w={"units": "ft**2"},
    AR={"units": "unitless"},
    taper={"units": "unitless"},
    S_h={"units":"ft**2"},
    fuse_length={"units":"ft"},
    Vh={"units":"unitless"}
)

prob.model.add_subsystem(
    "Vh",
    Vh_comp,
    promotes_inputs=[("S_w", av.Aircraft.Wing.AREA),
                     ('AR',av.Aircraft.Wing.ASPECT_RATIO),
                     ('taper',av.Aircraft.Wing.TAPER_RATIO),
                     ('S_h',av.Aircraft.HorizontalTail.AREA),
                     ('fuse_length',av.Aircraft.Fuselage.LENGTH)],
    promotes_outputs=["Vh"],
)

# Wing loading
# Boeing 737-9 takeoff wing loading
W_S_737 = 142.5  # lb/ft^2

ws_comp = om.ExecComp(
    "wing_loading = MTOW / area",
    MTOW={"units": "lbm"},
    area={"units": "ft**2"},
    wing_loading={"units": "lbm/ft**2"},
)

prob.model.add_subsystem(
    "wing_loading",
    ws_comp,
    promotes_inputs=[("area", av.Aircraft.Wing.AREA),
                     ('MTOW',av.Mission.Design.GROSS_MASS)],
    promotes_outputs=["wing_loading"],
)

prob.model.add_constraint("wing_loading", upper=W_S_737)
prob.model.add_constraint('Vh', lower=Vh_min, upper=Vh_max)

prob.add_objective()
prob.setup()
prob.set_val('wing_loading',140.0)
prob.run_aviary_problem(make_plots=make_plots)

print("\n=== AVIARY GEOMETRY OPTIMIZATION COMPLETE ===\n")

# Geometry Outputs
MTOW = prob.get_val(av.Mission.Design.GROSS_MASS)[0]
AR = prob.get_val(av.Aircraft.Wing.ASPECT_RATIO)
S = prob.get_val(av.Aircraft.Wing.AREA)
taper = prob.get_val(av.Aircraft.Wing.TAPER_RATIO)
sweep = prob.get_val(av.Aircraft.Wing.SWEEP)
fuselage_length = prob.get_val(av.Aircraft.Fuselage.LENGTH)
fuselage_width = prob.get_val(av.Aircraft.Fuselage.MAX_HEIGHT)
fuselage_height = prob.get_val(av.Aircraft.Fuselage.MAX_WIDTH)
total_fuel_burn = prob.get_val(av.Mission.Summary.FUEL_BURNED)

# Derived geometric quantities for OAS
span = np.sqrt(AR * S)           
c_avg = S / span                    
mach = 0.78
alt_ft = 35000.0
alt_m = alt_ft * 0.3048
T = 218.8
a = np.sqrt(1.4 * 287.05 * T)
rho = 0.38
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
    "fuselage wetted area [ft^2]": prob.get_val(av.Aircraft.Fuselage.WETTED_AREA),
    "H tail area [ft^2]": prob.get_val(av.Aircraft.HorizontalTail.AREA),
    "H tail aspect ratio": prob.get_val(av.Aircraft.HorizontalTail.ASPECT_RATIO),
    "H tail taper ratio": prob.get_val(av.Aircraft.HorizontalTail.TAPER_RATIO),
    "H tail wetted area": prob.get_val(av.Aircraft.HorizontalTail.WETTED_AREA),
    "V tail area [ft^2]": prob.get_val(av.Aircraft.VerticalTail.AREA),
    "V tail aspect ratio": prob.get_val(av.Aircraft.VerticalTail.ASPECT_RATIO),
    "V tail taper ratio": prob.get_val(av.Aircraft.VerticalTail.TAPER_RATIO),
    "V tail wetted area": prob.get_val(av.Aircraft.VerticalTail.WETTED_AREA)
}

print("=== REQUIRED OAS STRUCTURAL INPUTS ===")
for k, v in outputs.items():
    print(f"{k:20s} : {v}")

print("\nSaved. You can now pass these into your OAS structural optimization script.\n")

# Emissions Estimation
from emissions import compute_emissions
emissions = compute_emissions(prob, total_fuel_burn, make_plots=True)

# Compute Strucutre
#from structure import compute_structure
#structural_results = compute_structure(oas_inputs,plot_results=False)

#from carpet_plot import carpet_plot
"""
carpet_plot(prob,
            x_var=av.Aircraft.Wing.AREA,
            y_var=av.Aircraft.Wing.ASPECT_RATIO,
            output_var='Vh',
            x_range=(500, 2000),
            y_range=(0, 50),
            xlabel='Wing Area [ft^2]',
            ylabel='Aspect Ratio',
            title='Fuel Burn Carpet Plot')
"""