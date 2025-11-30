import aviary.api as av
from aviary.variable_info.variables import Aircraft as av_Aircraft

AviaryAircraft = av_Aircraft

class Aircraft(AviaryAircraft):
   
    class Cost:

        FLYAWAY = "aircraft:cost:flyaway"

        CONSUMER_PRICE = "aircraft:cost:consumer_price"

        MANUFACTURING_LABOR = "aircraft:cost:manufacturing_labor"

        MANUFACTURING_OVERHEAD = "aircraft:cost:manufacturing_overhead"

        MANUFACTURING_MATERIAL = "aircraft:cost:manufacturing_material"

        AIRFRAME_MANUFACTURING = "aircraft:cost:airframe_manufacturing"

        ENGINE_TOTAL = "aircraft:cost:engine_total"

        PROPULSION_SYSTEM = "aircraft:cost:propulsion_system"

        OTHER_EQUIPMENT = "aircraft:cost:other_equipment"

        DIRECT_MANUFACTURING = "aircraft:cost:direct_manufacturing"

        GA_OVERHEAD = "aircraft:cost:ga_overhead"

        TOTAL_MANUFACTURING = "aircraft:cost:total_manufacturing"

        DEALER = "aircraft:cost:dealer"

        OPTIONAL_EQUIPMENT = "aircraft:cost:optional_equipment"

        # Operating cost outputs
        # ------------------------------------------------------------------

        VAR_FUEL_OIL_PER_HR = "aircraft:cost:variable_fuel_oil_per_hr"

        VAR_INSPECTION_PER_HR = "aircraft:cost:variable_inspection_per_hr"

        VAR_OVERHAUL_PER_HR = "aircraft:cost:variable_overhaul_per_hr"

        VAR_MISC_PER_HR = "aircraft:cost:variable_misc_per_hr"

        VAR_TOTAL_PER_HR = "aircraft:cost:variable_total_per_hr"

        FIXED_STORAGE_ANNUAL = "aircraft:cost:fixed_storage_annual"

        FIXED_INSURANCE_ANNUAL = "aircraft:cost:fixed_insurance_annual"

        FIXED_DEPRECIATION_ANNUAL = "aircraft:cost:fixed_depreciation_annual"

        FIXED_FINANCING_TAX_ANNUAL = "aircraft:cost:fixed_financing_tax_annual"

        FIXED_CREW_ANNUAL = "aircraft:cost:fixed_crew_annual"

        FIXED_FAA_TAX_ANNUAL = "aircraft:cost:fixed_faa_tax_annual"

        FIXED_TOTAL_ANNUAL = "aircraft:cost:fixed_total_annual"

        TOTAL_COST_PER_HR = "aircraft:cost:total_cost_per_hr"

        MANUFACTURING_LABOR_HOURS = "aircraft:cost:manufacturing_labor_hours"

        # Inflation factors must be set by the user
        # ------------------------------------------------------------------

        LABOR_RATE_MFG = "aircraft:cost:input:labor_rate_mfg"

        LABOR_FACTOR = "aircraft:cost:input:labor_factor"

        LABOR_PROD_FACTOR = "aircraft:cost:input:labor_production_factor"

        MATERIAL_ADV_FACTOR = "aircraft:cost:input:material_advance_factor"

        MATERIAL_PROD_FACTOR = "aircraft:cost:input:material_production_factor"

        ENGINE_TYPE_CODE = "aircraft:cost:input:engine_type_code"

        TURBOCHARGE_FLAG = "aircraft:cost:input:turbocharge_flag"

        FUEL_PRICE_PER_GAL = "aircraft:cost:input:fuel_price_per_gal"

        OIL_PRICE_PER_GAL = "aircraft:cost:input:oil_price_per_gal"

        INSPECTION_COST = "aircraft:cost:input:inspection_cost"

        INSPECTION_INTERVAL_HR = "aircraft:cost:input:inspection_interval_hr"

        OVERHAUL_RATE = "aircraft:cost:input:overhaul_rate"

        OVERHAUL_INTERVAL_HR = "aircraft:cost:input:overhaul_interval_hr"

        VAR_MISC_PER_HR_INPUT = "aircraft:cost:input:variable_misc_per_hr"

        STORAGE_PER_MONTH = "aircraft:cost:input:storage_per_month"

        LIABILITY_INSURANCE = "aircraft:cost:input:liability_insurance"

        HULL_INSURANCE_RATE = "aircraft:cost:input:hull_insurance_rate"

        RESIDUAL_VALUE_FRACTION = "aircraft:cost:input:residual_value_fraction"

        DEPRECIATION_YEARS = "aircraft:cost:input:depreciation_years"

        OTHER_FIXED_EXPENSE = "aircraft:cost:input:other_fixed_expense"

        LOAN_INTEREST_RATE = "aircraft:cost:input:loan_interest_rate"

        PROPERTY_TAX_RATE = "aircraft:cost:input:property_tax_rate"

        CREW_COST_BASE = "aircraft:cost:input:crew_cost_base_annual"

        CREW_OVERHEAD_FRACTION = "aircraft:cost:input:crew_overhead_fraction"

        UTILIZATION_ANNUAL = "aircraft:cost:input:utilization_annual_hours"

        OPTIONAL_EQUIP_FLAG = "aircraft:cost:input:optional_equip_flag"

ExtendedMetaData = av.CoreMetaData

def _add(name, desc, default_value=0.0):
    
    av.add_meta_data(name,units=None, desc=desc,default_value=default_value,meta_data=ExtendedMetaData,)
    
_add(Aircraft.Cost.FLYAWAY,"Flyaway factory price",)

_add(Aircraft.Cost.CONSUMER_PRICE,"Total consumer price",)

# Manufacturing cost 
#-------------------------------------------------------------

_add(Aircraft.Cost.MANUFACTURING_LABOR,"MFG labor cost",)

_add(Aircraft.Cost.MANUFACTURING_OVERHEAD,"MFG labor overhead cost",)

_add(Aircraft.Cost.MANUFACTURING_MATERIAL,"MFG materials cost",)

_add(Aircraft.Cost.AIRFRAME_MANUFACTURING,"Total airframe MFG cost",)

_add(Aircraft.Cost.ENGINE_TOTAL,"Total engine cost",)

_add(Aircraft.Cost.PROPULSION_SYSTEM,"Total propulsion cost",)

_add(Aircraft.Cost.OTHER_EQUIPMENT,"AUX equipment cost",)

_add(Aircraft.Cost.DIRECT_MANUFACTURING,"Direct MFG cost",)

_add(Aircraft.Cost.GA_OVERHEAD,"administrative cost",)

_add(Aircraft.Cost.TOTAL_MANUFACTURING,"Total MFG cost",)

_add(Aircraft.Cost.DEALER,"Aircraft purchase price",)

_add(Aircraft.Cost.OPTIONAL_EQUIPMENT,"Optional equipment cost",)

# Operational costs
#---------------------------------------------------------------------------------------

_add(Aircraft.Cost.VAR_FUEL_OIL_PER_HR,"Fuel and oil variable cost per flight hour",)

_add(Aircraft.Cost.VAR_INSPECTION_PER_HR,"Inspection and basic maintenance cost per flight hour",)

_add(Aircraft.Cost.VAR_OVERHAUL_PER_HR,"Engine overhaul cost per flight hour",)

_add(Aircraft.Cost.VAR_MISC_PER_HR,"Other variable operating cost per flight hour",)

_add(Aircraft.Cost.VAR_TOTAL_PER_HR,"Total variable operating cost per flight hour",)

_add(Aircraft.Cost.FIXED_STORAGE_ANNUAL,"Annual storage cost",)

_add(Aircraft.Cost.FIXED_INSURANCE_ANNUAL,"Annual insurance cost",)

_add(Aircraft.Cost.FIXED_DEPRECIATION_ANNUAL,"Annual depreciation expense",)

_add(Aircraft.Cost.FIXED_FINANCING_TAX_ANNUAL,"Annual loan, interest, and tax fixed costs",)

_add(Aircraft.Cost.FIXED_CREW_ANNUAL,"Annual crew cost including overhead",)

_add(Aircraft.Cost.FIXED_FAA_TAX_ANNUAL,"Annual FAA weight-based tax",)

_add(Aircraft.Cost.FIXED_TOTAL_ANNUAL,"Total fixed operating cost",)

_add(Aircraft.Cost.TOTAL_COST_PER_HR,"Total operating cost per flight hour",)

_add(Aircraft.Cost.MANUFACTURING_LABOR_HOURS,"Manufacturing labor hours",)

# External cost parameters
#-------------------------------------------------------------------------------------------
_add(Aircraft.Cost.LABOR_RATE_MFG,"MFG labor rate",default_value=3.40,)

_add(Aircraft.Cost.LABOR_FACTOR,"Labor constant",default_value=1.0,)

_add(Aircraft.Cost.LABOR_PROD_FACTOR,"Labor production constant",default_value=1.0,)

_add(Aircraft.Cost.MATERIAL_ADV_FACTOR,"Material advancement factor",default_value=0.0,)

_add(Aircraft.Cost.MATERIAL_PROD_FACTOR,"Material production factor",default_value=1.0,)

_add(Aircraft.Cost.ENGINE_TYPE_CODE,"NTYE",default_value=7,)

_add(Aircraft.Cost.TURBOCHARGE_FLAG,"Turbocharging constant",default_value=0,)

_add(Aircraft.Cost.FUEL_PRICE_PER_GAL,"Fuel price per gallon",default_value=0.0,)

_add(Aircraft.Cost.OIL_PRICE_PER_GAL,"Oil price per gallon",default_value=2.0,)

_add(Aircraft.Cost.INSPECTION_COST,"Inspection cost",default_value=0.0,)

_add(Aircraft.Cost.INSPECTION_INTERVAL_HR,"Hours btw inspections",default_value=0.0,)

_add(Aircraft.Cost.OVERHAUL_RATE,"Engine overhaul rate",default_value=0.0,)

_add(Aircraft.Cost.OVERHAUL_INTERVAL_HR,"Time btw engine overhauls",default_value=0.0,)

_add(Aircraft.Cost.VAR_MISC_PER_HR_INPUT,"Misc operating costs per flight hour",default_value=0.0,)

_add(Aircraft.Cost.STORAGE_PER_MONTH,"Storage cost",default_value=0.0,)

_add(Aircraft.Cost.LIABILITY_INSURANCE,"Liability insurance cost",default_value=0.0,)

_add(Aircraft.Cost.HULL_INSURANCE_RATE,"Hull insurance rate",default_value=0.0,)

_add(Aircraft.Cost.RESIDUAL_VALUE_FRACTION,"Residual value",default_value=0.0,)

_add(Aircraft.Cost.DEPRECIATION_YEARS,"Depreciation years",default_value=10.0,)

_add(Aircraft.Cost.OTHER_FIXED_EXPENSE,"Other fixed expenses per year",default_value=0.0,)

_add(Aircraft.Cost.LOAN_INTEREST_RATE,"Loan interest rate",default_value=0.0,)

_add(Aircraft.Cost.PROPERTY_TAX_RATE,"Property tax rate",default_value=0.0,)

_add(Aircraft.Cost.CREW_COST_BASE,"Base crew cost per year",default_value=0.0,)

_add(Aircraft.Cost.CREW_OVERHEAD_FRACTION,"Crew overhead constant",default_value=0.0,)

_add(Aircraft.Cost.UTILIZATION_ANNUAL,"Annual utilization",default_value=500.0,)

_add(Aircraft.Cost.OPTIONAL_EQUIP_FLAG,"Optional equipment flag",default_value=0.0,)
