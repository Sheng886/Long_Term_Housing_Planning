
----------------------Parameter----------------------

[Distance]
I      : Supply Node Location	
W      : Transition Node Location	
J      : Study Region Location	

C_wjp  : w->j Transition Cost
C_iwp  : i->w Transition Cost
	

[House Info]
I_p    : Installation Cost of house p
R_p    : Recycled rate of house p 
CH_p   : Holding Cost
L_pa   : Attribute value of house p
O_p    : Acqusition Cost

[Study Region Info]
H_ag   : Attribute Weight of Victim g
Dk_gt  : Housing Demand 
CU_g   : Unmet Penalty Cost

[Transition Node Info]
U_w    : Capacity of Transition Node w
E_p    : Maximum Number of solution provide in a season (not stored item)

----------------------File---------------------------

demand 1-6		   		  : household 1-6 demand for each study region
House_Attribute_flood/wind 		  : household vs attribute matrix under flood/wind
House_Info		   		  : I_p, CH_p, O_p of each type house
Staging_Area/Study_Region/Supply_node_loc : longitude and latitude of Staging_Area/Study_Region/Supply_node
Victim_Info				  : CU_g of each type house
Victim_Weight				  : Attribute weight of each group of victim














