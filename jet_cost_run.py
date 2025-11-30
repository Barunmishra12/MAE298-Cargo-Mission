import aviary.api as av

from aviary.interface import reports as av_reports
def _dummy_input_check_report(*args, **kwargs):
    return
av_reports.input_check_report = _dummy_input_check_report

from jet_cost_variables import Aircraft
from jet_cost_builder import JetCostBuilder

if __name__ == '__main__':
   
    phase_info = {
    'pre_mission': {'include_takeoff': False, 'optimize_mass': False},
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

    phase_info["post_mission"]["external_subsystems"] = [JetCostBuilder()]

    prob = av.AviaryProblem()

    prob.load_inputs("aviary/models/aircraft/advanced_single_aisle/advanced_single_aisle_baseline_FLOPS.csv",phase_info,)

    prob.check_and_preprocess_inputs()

    prob.build_model()

    prob.add_driver("IPOPT")

    prob.add_design_variables()

    prob.model.add_objective(Aircraft.Cost.FLYAWAY)

    prob.setup()
    
    prob.set_val(Aircraft.Cost.LABOR_RATE_MFG, 100.0)
    prob.set_val(Aircraft.Cost.LABOR_FACTOR, 1.0)
    prob.set_val(Aircraft.Cost.LABOR_PROD_FACTOR, 1.0)
    prob.set_val(Aircraft.Cost.FUEL_PRICE_PER_GAL, 6.5)
    prob.set_val(Aircraft.Cost.UTILIZATION_ANNUAL, 8000)
    prob.set_val(Aircraft.Cost.RESIDUAL_VALUE_FRACTION, 0.15)
    prob.set_val(Aircraft.Cost.DEPRECIATION_YEARS, 20.0)
    prob.set_val(Aircraft.Cost.CREW_COST_BASE, 1000000.0)
    prob.set_val(Aircraft.Cost.CREW_OVERHEAD_FRACTION, 0.035)
    prob.set_val(Aircraft.Cost.STORAGE_PER_MONTH, 25000.0)
    prob.set_val(Aircraft.Cost.LIABILITY_INSURANCE, 200000.0)
    prob.set_val(Aircraft.Cost.HULL_INSURANCE_RATE, 0.01)
    prob.set_val(Aircraft.Cost.OVERHAUL_RATE, 1.0e-4)
    prob.set_val(Aircraft.Cost.OVERHAUL_INTERVAL_HR, 20000.0)

    prob.run_aviary_problem(suppress_solver_print=True)

    flyaway        = prob.get_val(Aircraft.Cost.FLYAWAY)[0]
    consumer_price = prob.get_val(Aircraft.Cost.CONSUMER_PRICE)[0]
    toc            = prob.get_val(Aircraft.Cost.TOTAL_COST_PER_HR)[0]

    var_cost       = prob.get_val(Aircraft.Cost.VAR_TOTAL_PER_HR)[0]
    fixed_cost     = prob.get_val(Aircraft.Cost.FIXED_TOTAL_ANNUAL)[0]

    airframe_man   = prob.get_val(Aircraft.Cost.AIRFRAME_MANUFACTURING)[0]
    engine_tot     = prob.get_val(Aircraft.Cost.ENGINE_TOTAL)[0]
    direct_man     = prob.get_val(Aircraft.Cost.DIRECT_MANUFACTURING)[0]
    ga_overhead    = prob.get_val(Aircraft.Cost.GA_OVERHEAD)[0]
    total_man      = prob.get_val(Aircraft.Cost.TOTAL_MANUFACTURING)[0]

    lab_hr         = prob.get_val(Aircraft.Cost.MANUFACTURING_LABOR_HOURS)[0]
    lab_cost       = prob.get_val(Aircraft.Cost.MANUFACTURING_LABOR)[0]
    overhaul       = prob.get_val(Aircraft.Cost.VAR_OVERHAUL_PER_HR)[0]
    aux_systems    = prob.get_val(Aircraft.Cost.OPTIONAL_EQUIPMENT)[0]

    crew_annual    = prob.get_val(Aircraft.Cost.FIXED_CREW_ANNUAL)[0]

    # For fuel uplift cost
    burnt_fuel_cost = prob.get_val(Aircraft.Cost.VAR_FUEL_OIL_PER_HR)[0]
    fuel_mass      = prob.get_val("mission:summary:total_fuel_mass")[0]      
    fuel_density   = prob.get_val("aircraft:fuel:density")[0]               
    fuel_price     = prob.get_val(Aircraft.Cost.FUEL_PRICE_PER_GAL)[0] 

    print("\n==================== COST RESULTS ====================\n")

    # ----- Labor Summary -----
    print("----- Labor Summary -----")
    print(f"Manufacturing labor hours:       {lab_hr:,.3f} hr")
    print(f"Manufacturing labor cost:        {lab_cost:,.3f} USD")

    # ----- Manufacturing Breakdown -----
    print("\n----- Manufacturing Breakdown -----")
    print(f"Airframe manufacturing:          {airframe_man:,.3f} USD")
    print(f"Engine total:                    {engine_tot:,.3f} USD")
    print(f"Auxiliary systems (avionics/APU):{aux_systems:,.3f} USD")
    print(f"Direct manufacturing cost:       {direct_man:,.3f} USD")
    print(f"General & admin overhead:        {ga_overhead:,.3f} USD")
    print(f"Total manufacturing cost:        {total_man:,.3f} USD")

        # ----- Price & Total Economics -----
    print("\n----- Price & Total Economics -----")
    print(f"Base aircraft price (flyaway):   {flyaway:,.3f} USD")
    print(f"Final aircraft price (consumer): {consumer_price:,.3f} USD")
    

    # ----- Operating Cost Items -----
    print("\n----- Operating Cost Items -----")
    print(f"Total uplifted fuel cost:        {(fuel_mass/fuel_density)*fuel_price:,.3f} USD")
    print(f"Fuel cost per mission:           {burnt_fuel_cost:,.3f} USD")
    print(f"Total operating cost:            {toc:,.3f} USD/hr")
    print(f"Annual crew cost:                {crew_annual:,.3f} USD/year")
    print(f"Total fixed operating cost:      {fixed_cost:,.3f} USD/year")
    


    print("\n======================================================\n")


