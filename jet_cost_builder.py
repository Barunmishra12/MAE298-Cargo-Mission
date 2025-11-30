import openmdao.api as om

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.utils.aviary_values import AviaryValues

from jet_cost import JetCost
from jet_cost_variables import Aircraft

class JetCostBuilder(SubsystemBuilderBase):

    def __init__(self, name: str = "jet_cost"):

        super().__init__(name)

    def build_post_mission(self,aviary_inputs: AviaryValues,**kwargs,): 

        cost_group = om.Group()

        cost_group.add_subsystem("jet_cost_comp",JetCost(),
                                 
            promotes_inputs=["aircraft:*","mission:*",],
            
            promotes_outputs=[Aircraft.Cost.FLYAWAY,Aircraft.Cost.CONSUMER_PRICE,Aircraft.Cost.TOTAL_COST_PER_HR,Aircraft.Cost.VAR_TOTAL_PER_HR,
        Aircraft.Cost.FIXED_TOTAL_ANNUAL,Aircraft.Cost.AIRFRAME_MANUFACTURING,Aircraft.Cost.ENGINE_TOTAL,Aircraft.Cost.DIRECT_MANUFACTURING,
        Aircraft.Cost.GA_OVERHEAD,Aircraft.Cost.TOTAL_MANUFACTURING,Aircraft.Cost.MANUFACTURING_LABOR_HOURS,Aircraft.Cost.MANUFACTURING_LABOR,
        Aircraft.Cost.VAR_OVERHAUL_PER_HR,Aircraft.Cost.OPTIONAL_EQUIPMENT,Aircraft.Cost.FIXED_CREW_ANNUAL,Aircraft.Cost.VAR_FUEL_OIL_PER_HR,],

        )
        return cost_group
