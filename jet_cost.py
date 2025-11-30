import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Mission
from jet_cost_variables import Aircraft

class JetCost(om.ExplicitComponent):

    def setup(self):

        #Avairy Pre and post mission inputs
        self.add_input("aircraft:design:empty_mass",desc="Empty weight",units="lbm",)
        self.add_input(Mission.Design.GROSS_MASS,desc="Design gross mass",units="lbm",)
        self.add_input("aircraft:engine:num_engines",desc="Number of engines",units=None,)
        self.add_input("aircraft:engine:scaled_sls_thrust",desc="Sea-level static thrust per engine",units="lbf",val=0.0,)
        self.add_input("mission:summary:total_fuel_mass",desc="Total fuel weight",units="lbm",)
        self.add_input("mission:summary:total_fuel_burn",desc="Fuel weight used per mission block",units="lbm",)
        self.add_input("block_time",desc="Block flight time ",units="h",val=4.5,)
        self.add_input("aircraft:fuel:density",desc="Fuel density",units="lbm/galUS",)
        self.add_input("mission:cruise:mach",desc="mach at cruise",units= None,)
        
        # Labor & materials
        self.add_input(Aircraft.Cost.LABOR_RATE_MFG)    
        self.add_input(Aircraft.Cost.LABOR_FACTOR)            
        self.add_input(Aircraft.Cost.LABOR_PROD_FACTOR)       
        self.add_input(Aircraft.Cost.MATERIAL_ADV_FACTOR)     
        self.add_input(Aircraft.Cost.MATERIAL_PROD_FACTOR)    

        # Variable operating cost parameters
        self.add_input(Aircraft.Cost.FUEL_PRICE_PER_GAL)
        self.add_input(Aircraft.Cost.OIL_PRICE_PER_GAL)
        self.add_input(Aircraft.Cost.INSPECTION_COST)
        self.add_input(Aircraft.Cost.INSPECTION_INTERVAL_HR)
        self.add_input(Aircraft.Cost.OVERHAUL_RATE)
        self.add_input(Aircraft.Cost.OVERHAUL_INTERVAL_HR)
        self.add_input(Aircraft.Cost.VAR_MISC_PER_HR_INPUT)

        # Fixed operating cost parameters
        self.add_input(Aircraft.Cost.STORAGE_PER_MONTH)
        self.add_input(Aircraft.Cost.LIABILITY_INSURANCE)
        self.add_input(Aircraft.Cost.HULL_INSURANCE_RATE)
        self.add_input(Aircraft.Cost.RESIDUAL_VALUE_FRACTION)
        self.add_input(Aircraft.Cost.DEPRECIATION_YEARS)
        self.add_input(Aircraft.Cost.OTHER_FIXED_EXPENSE)
        self.add_input(Aircraft.Cost.LOAN_INTEREST_RATE)
        self.add_input(Aircraft.Cost.PROPERTY_TAX_RATE)
        self.add_input(Aircraft.Cost.CREW_COST_BASE)
        self.add_input(Aircraft.Cost.CREW_OVERHEAD_FRACTION)
        self.add_input(Aircraft.Cost.UTILIZATION_ANNUAL)
        self.add_input(Aircraft.Cost.OPTIONAL_EQUIP_FLAG)

        # Outputs
        # ------------------------------------------------------------------

        self.add_output(Aircraft.Cost.MANUFACTURING_LABOR_HOURS)
        self.add_output(Aircraft.Cost.MANUFACTURING_LABOR)
        self.add_output(Aircraft.Cost.MANUFACTURING_OVERHEAD)
        self.add_output(Aircraft.Cost.MANUFACTURING_MATERIAL)
        self.add_output(Aircraft.Cost.AIRFRAME_MANUFACTURING)
        self.add_output(Aircraft.Cost.ENGINE_TOTAL)
        self.add_output(Aircraft.Cost.OTHER_EQUIPMENT)
        self.add_output(Aircraft.Cost.PROPULSION_SYSTEM)
        self.add_output(Aircraft.Cost.DIRECT_MANUFACTURING)
        self.add_output(Aircraft.Cost.GA_OVERHEAD)
        self.add_output(Aircraft.Cost.TOTAL_MANUFACTURING)
        self.add_output(Aircraft.Cost.DEALER)
        self.add_output(Aircraft.Cost.OPTIONAL_EQUIPMENT)
        self.add_output(Aircraft.Cost.FLYAWAY)
        self.add_output(Aircraft.Cost.CONSUMER_PRICE)

        # Variable operating costs
        self.add_output(Aircraft.Cost.VAR_FUEL_OIL_PER_HR)
        self.add_output(Aircraft.Cost.VAR_INSPECTION_PER_HR)
        self.add_output(Aircraft.Cost.VAR_OVERHAUL_PER_HR)
        self.add_output(Aircraft.Cost.VAR_MISC_PER_HR)
        self.add_output(Aircraft.Cost.VAR_TOTAL_PER_HR)

        # Fixed operating costs (annual)
        self.add_output(Aircraft.Cost.FIXED_STORAGE_ANNUAL)
        self.add_output(Aircraft.Cost.FIXED_INSURANCE_ANNUAL)
        self.add_output(Aircraft.Cost.FIXED_DEPRECIATION_ANNUAL)
        self.add_output(Aircraft.Cost.FIXED_FINANCING_TAX_ANNUAL)
        self.add_output(Aircraft.Cost.FIXED_CREW_ANNUAL)
        self.add_output(Aircraft.Cost.FIXED_FAA_TAX_ANNUAL)
        self.add_output(Aircraft.Cost.FIXED_TOTAL_ANNUAL)

        # Total cost per flight hour
        self.add_output(Aircraft.Cost.TOTAL_COST_PER_HR)

        
        self.declare_partials(of="*", wrt="*", method="fd")


    def compute(self, inputs, outputs):
        
        # Basic physical quantities
        # ------------------------------
        W_EMP  = float(inputs["aircraft:design:empty_mass"])
        W_G    = float(inputs[Mission.Design.GROSS_MASS])
        M      = float(inputs["mission:cruise:mach"])
        EN     = float(inputs["aircraft:engine:num_engines"])
        T_SLS  = float(inputs["aircraft:engine:scaled_sls_thrust"])

        A_LR   = float(inputs[Aircraft.Cost.LABOR_RATE_MFG])
        C_LP   = float(inputs[Aircraft.Cost.LABOR_FACTOR])
        PROD_PL = float(inputs[Aircraft.Cost.LABOR_PROD_FACTOR])
        ADV_MP = float(inputs[Aircraft.Cost.MATERIAL_ADV_FACTOR])
        PROD_PM = float(inputs[Aircraft.Cost.MATERIAL_PROD_FACTOR])
      
        a = 1116.45  # speed of sound 
        V_ft_s = M * a
        WSP = W_EMP * (V_ft_s / 1000.0)

        # Manufacturing, labor & material cost 
        # --------------------------------------

        W_EMP_K = W_EMP / 1000.0

        # MFG labor hours
        D_MLH = 1200.0 * W_EMP_K    
        outputs[Aircraft.Cost.MANUFACTURING_LABOR_HOURS] = D_MLH

        # Labor cost
        CS_ML = D_MLH * A_LR * C_LP * PROD_PL
        outputs[Aircraft.Cost.MANUFACTURING_LABOR] = CS_ML

        # Material cost
        CS_MM = 1500.0 * W_EMP_K     
        CS_MM *= (1.0 + ADV_MP) * PROD_PM
        outputs[Aircraft.Cost.MANUFACTURING_MATERIAL] = CS_MM

        # Overhead cost
        CS_OH = 1.30 * CS_ML
        outputs[Aircraft.Cost.MANUFACTURING_OVERHEAD] = CS_OH

        # Airframe MFG cost
        CS_AFF = CS_ML + CS_MM + CS_OH
        outputs[Aircraft.Cost.AIRFRAME_MANUFACTURING] = CS_AFF

        # Engine cost
        # ------------------------------

        CS_ENG_total = 2* EN * 1100 * (T_SLS ** 0.88)
        outputs[Aircraft.Cost.ENGINE_TOTAL] = CS_ENG_total

        # Auxilary engine equipments
        CS_OEQ = 9.6e-7 * (WSP ** 1.698)
        outputs[Aircraft.Cost.OTHER_EQUIPMENT] = CS_OEQ

        # Total Engine cost:
        CS_TEQ = CS_ENG_total + CS_OEQ
        outputs[Aircraft.Cost.PROPULSION_SYSTEM] = CS_TEQ

        # Aircraft manufacturing total 
        CS_DMF = CS_AFF + CS_TEQ
        outputs[Aircraft.Cost.DIRECT_MANUFACTURING] = CS_DMF

        # Profit calculations
        CS_GA = 0.167 * (W_EMP ** 0.8743)
        outputs[Aircraft.Cost.GA_OVERHEAD] = CS_GA

        # Final total manufacturing cost
        CS_MANF = CS_DMF + CS_GA
        outputs[Aircraft.Cost.TOTAL_MANUFACTURING] = CS_MANF

        # Manufacturer profit fraction:
        PROF_G = 0.066 + 2.33e-5 * W_EMP
        if PROF_G > 0.18:
            PROF_G = 0.18

        # Dealer cost:
        #   CS_DLR = CS_MANF * (1 + PROF_G)
        CS_DLR = CS_MANF * (1.0 + PROF_G)
        outputs[Aircraft.Cost.DEALER] = CS_DLR

        # Distributor markup fraction:
        DD_MARK = 0.1695 * (W_EMP ** 0.8743)
        if DD_MARK > 0.30:
            DD_MARK = 0.30

        # Airline Accquisiton cost:
        CS_FAF = CS_DLR * (1.0 + DD_MARK)
        outputs[Aircraft.Cost.FLYAWAY] = CS_FAF

        # Optional equipment cost
        N_CADE = inputs[Aircraft.Cost.OPTIONAL_EQUIP_FLAG]
        if N_CADE != 0.0 and CS_FAF > 0.0:
            CADEL = 1.015 * np.log10(CS_FAF) - 0.70782
            C_ADE = 10.0 ** CADEL
        else:
            C_ADE = 0.0
        outputs[Aircraft.Cost.OPTIONAL_EQUIPMENT] = C_ADE

        # Updated Acquistion cost
        CP = CS_FAF + C_ADE
        outputs[Aircraft.Cost.CONSUMER_PRICE] = CP

        # Operational cost(Variable)
        # ------------------------------
        fuel_price = inputs[Aircraft.Cost.FUEL_PRICE_PER_GAL]
        oil_price = inputs[Aircraft.Cost.OIL_PRICE_PER_GAL]
        C_INF = inputs[Aircraft.Cost.INSPECTION_COST]
        HR_I = inputs[Aircraft.Cost.INSPECTION_INTERVAL_HR]
        OH_R = inputs[Aircraft.Cost.OVERHAUL_RATE]
        TB_O = inputs[Aircraft.Cost.OVERHAUL_INTERVAL_HR]
        C_MV_in = inputs[Aircraft.Cost.VAR_MISC_PER_HR_INPUT]

        W_f = inputs["mission:summary:total_fuel_burn"]
        ST = inputs["block_time"]
        F_wtf = inputs["aircraft:fuel:density"]

        # Fuel & oil cost gallon per hour
        F_CON = W_f / (ST * F_wtf)

        O_CON = 0.135 * EN * (4.0 / 8.1)

        if oil_price <= 0.0:
                oil_price_eff = 2.0
        else:
                oil_price_eff = oil_price
        FOC = ((F_CON * fuel_price) + (O_CON * oil_price_eff))*10000

        outputs[Aircraft.Cost.VAR_FUEL_OIL_PER_HR] = FOC

        # Inspection cost per hour:
        if HR_I > 0.0:
            A_IC = C_INF / HR_I
        else:
            A_IC = 0.0
        outputs[Aircraft.Cost.VAR_INSPECTION_PER_HR] = A_IC

        # Engine overhaul cost per hour 
        if TB_O > 0.0 and OH_R > 0.0:
            OH_C = EN * T_SLS * (OH_R / TB_O)
        else:
            OH_C = 0.0
        outputs[Aircraft.Cost.VAR_OVERHAUL_PER_HR] = OH_C

        # Other variable operating cost per hour:
        outputs[Aircraft.Cost.VAR_MISC_PER_HR] = C_MV_in

        # Total variable cost per hour:
        C_VAR = FOC + A_IC + OH_C + C_MV_in
        outputs[Aircraft.Cost.VAR_TOTAL_PER_HR] = C_VAR

        # ------------------------------
        # Fixed operating costs (annual)
        # ------------------------------

        SR_PM = inputs[Aircraft.Cost.STORAGE_PER_MONTH]
        C_LIAB = inputs[Aircraft.Cost.LIABILITY_INSURANCE]
        H_IR = inputs[Aircraft.Cost.HULL_INSURANCE_RATE]
        PR_v = inputs[Aircraft.Cost.RESIDUAL_VALUE_FRACTION]
        D_YR = inputs[Aircraft.Cost.DEPRECIATION_YEARS]
        CMP = inputs[Aircraft.Cost.OTHER_FIXED_EXPENSE]
        R_I = inputs[Aircraft.Cost.LOAN_INTEREST_RATE]
        T_R = inputs[Aircraft.Cost.PROPERTY_TAX_RATE]
        C_CRW_base = inputs[Aircraft.Cost.CREW_COST_BASE]
        CRW_OH = inputs[Aircraft.Cost.CREW_OVERHEAD_FRACTION]
        U_annual = inputs[Aircraft.Cost.UTILIZATION_ANNUAL]

        # Storage cost per year
        SC = 12.0 * SR_PM
        outputs[Aircraft.Cost.FIXED_STORAGE_ANNUAL] = SC

        # Insurance cost per year
        C_I = C_LIAB + H_IR * CP
        outputs[Aircraft.Cost.FIXED_INSURANCE_ANNUAL] = C_I

        # Depreciation
        if D_YR > 0.0 and PR_v >= 0.0:
            DEP = CP * (1.0 - (PR_v ** (1.0 / D_YR)))
        else:
            DEP = 0.0
        outputs[Aircraft.Cost.FIXED_DEPRECIATION_ANNUAL] = DEP

        # Tax estimates
        T_IC = 0.80 * CP * R_I       
        T_C = T_R * CP      
        C_FO = T_IC + T_C + CMP   
        outputs[Aircraft.Cost.FIXED_FINANCING_TAX_ANNUAL] = C_FO

        # cost of crew
        C_CRW = C_CRW_base * (1.0 + CRW_OH)
        outputs[Aircraft.Cost.FIXED_CREW_ANNUAL] = C_CRW

        # FAA tax
        FAA_tax = 25.0 + (0.035 * W_G)
        outputs[Aircraft.Cost.FIXED_FAA_TAX_ANNUAL] = FAA_tax

        # Total fixed cost per year:
        C_FIX = SC + C_I + DEP + C_FO + C_CRW + FAA_tax
        outputs[Aircraft.Cost.FIXED_TOTAL_ANNUAL] = C_FIX

        # Operational cost per flight hour
        TOC = (C_VAR + (C_FIX / U_annual))
        outputs[Aircraft.Cost.TOTAL_COST_PER_HR] = TOC
