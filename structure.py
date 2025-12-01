import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.structures.struct_groups import SpatialBeamAlone

def compute_structure(oas_inputs, plot_results=False):
    """
    Runs an OpenAeroStruct structural optimization and returns results
    in a dictionary, callable from external scripts.

    Parameters
    ----------
    oas_inputs : dict
        Dictionary containing geometry + load inputs.
    plot_results : bool
        Whether to generate plots.

    Returns
    -------
    results : dict
        Dictionary containing optimized thickness, structural mass, failures, etc.
    """

    # -------------------- DEFAULT INPUTS --------------------
    g = 9.80665
    AR = float(oas_inputs["aspect_ratio"])
    S = float(oas_inputs["wing_area"])
    span = float(oas_inputs["span"])
    c_avg = float(oas_inputs["mean_chord"])
    taper = float(oas_inputs["taper_ratio"])
    sweep_deg = float(oas_inputs["sweep_deg"])
    MTOW = float(oas_inputs["aircraft_mass"])
    rho = float(oas_inputs["rho"])
    load_factor = float(oas_inputs["load_factor"])

    # -------------------- mesh setup --------------------
    mesh_dict = {
        "num_x": 3,                 # chordwise panels (small for structural-only)
        "num_y": 17,                # spanwise panels
        "wing_type": "CRM",         # Trapezoidal Planform
        "symmetry": True, 
        "num_twist_cp":5            
    }
    mesh, twist_cp = generate_mesh(mesh_dict)

    # -------------------- surface / structural dictionary --------------------
    surf_dict = {
        "name": "wing",
        "symmetry": True,
        "fem_model_type": "tube",   # simple spar-like model (same as OAS example)
        "mesh": mesh,
        # Material properties (aluminum-ish baseline)
        "E": 70.0e9,                # Pa (Young's modulus)
        "G": 30.0e9,                # Pa (shear modulus)
        "yield": 500.0e6,           # Pa (yield)
        "safety_factor": 2.5,       # factor used in failure check
        "mrho": 3.0e3,              # kg/m^3 material density
        "fem_origin": 0.35,         # normalized chordwise location of the spar
        # thickness-to-chord and initial thickness control points (B-spline control points)
        "t_over_c_cp": np.array([0.12]),   # baseline t/c for wingbox airfoil
        # thickness_cp controls the structural thickness design vars (one or more CPs)
        "thickness_cp": np.ones((3,)) * 0.02,  # initial thicknesses (m) - tuned for optimizer
        "wing_weight_ratio": 2.0,
        "struct_weight_relief": False,
        "distributed_fuel_weight": False,
        "exact_failure_constraint": False,
    }

    # -------------------- build OpenMDAO problem --------------------
    prob = om.Problem()
    ny = surf_dict["mesh"].shape[1]

    indep_var_comp = om.IndepVarComp()

    # Total maneuver load
    total_vertical_load = MTOW * g * load_factor  # N 
    half_span_vertical_load = total_vertical_load / 2.0  # N

    loads = np.zeros((ny, 6))
    loads[:, 2] = half_span_vertical_load / float(ny)  # N per node

    # Load factor and loads inputs
    indep_var_comp.add_output("loads", val=loads, units="N")
    indep_var_comp.add_output("load_factor", val=load_factor)

    struct_group = SpatialBeamAlone(surface=surf_dict)
    struct_group.add_subsystem("indep_vars", indep_var_comp, promotes=["*"])
    prob.model.add_subsystem(surf_dict["name"], struct_group)

    # -------------------- driver / optimizer --------------------
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["disp"] = True
    prob.driver.options["tol"] = 1e-6

    #   -------------------- design variables, constraints, objective --------------------
    prob.model.add_design_var("wing.thickness_cp", lower=0.005, upper=0.2, ref=1e-1)

    # Constraints
    prob.model.add_constraint("wing.failure", upper=0.0)
    #prob.model.add_constraint("wing.thickness_intersects", upper=0.0)

    # Objective: minimize structural mass
    prob.model.add_objective("wing.structural_mass", scaler=1e-5)

    # -------------------- setup & run --------------------
    prob.setup()
    # Run the optimizer
    prob.run_driver()

    # -------------------- outputs --------------------
    struct_mass = prob.get_val("wing.structural_mass")
    thickness_cp_opt = prob.get_val("wing.thickness_cp")
    failure = prob.get_val("wing.failure")
    wing_forces = prob.get_val("wing.forces")

    print("\n==== OpenAeroStruct Analysis Results ====")
    print(f"optimized thickness_cp  : {thickness_cp_opt.ravel()}")
    print(f"structural_mass (kg)    : {struct_mass.ravel()[0]:.3f}")
    print(f"max failure constraint  : {failure.ravel()}")
    # -------------------- PACKAGE OUTPUT --------------------
    results = {
        "thickness_cp": thickness_cp_opt,
        "structural_mass": struct_mass,
        "failure": failure,
        "wing_forces": wing_forces,
    }

    return results


def _plot_oas_results(spanwise_y, t_interp, failure, mesh, deformation):
    """Internal helper to plot results."""
    plt.figure()
    plt.plot(spanwise_y, t_interp, "o-")
    plt.xlabel("Spanwise y (m)")
    plt.ylabel("Thickness (m)")
    plt.title("Optimized Wingbox Thickness")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(spanwise_y, failure, "o-")
    plt.xlabel("Spanwise y (m)")
    plt.ylabel("Failure")
    plt.title("Structural Failure Distribution")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(mesh[0, :, 0], deformation[0, :, 2], "o-")
    plt.xlabel("Chordwise x (m)")
    plt.ylabel("Vertical Displacement (m)")
    plt.title("Wing Deformation")
    plt.grid(True)
    plt.show()
