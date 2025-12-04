# Introduction:
# This function recreates the geometry and mesh of the aircraft modelusing the data from the Aviary 
# input file (.csv).
"""'''
# Mention: This is not a ferfect conversion, only main components are created.
# For higher fidelity, you should have a mesh dependeence study to refine the mesh and also consider 
# the parameters listed below:

# The parameter lost in "Advanced Single Aisle" model are:
# Wing airfoil shape
# Wing, vertical tail and horizontal tail location details
# Wing twist details
"""

import csv
import math
import os

# Load OpenVSP Python bindings if available. Editors/linters may report
# "could not be resolved" when the bindings are not installed in the
# selected interpreter or when they are compiled/native extensions.
try:
    import openvsp as vsp  # type: ignore
except Exception:
    vsp = None

FT2M = 0.3048
FT2M2 = FT2M ** 2

def load_aviary_csv(path):
    """
    Load Aviary-style FLOPS CSV into a dictionary:
    data['aircraft:wing:span'] -> {'value': ..., 'units': ..., 'extra': [...]}
    """
    data = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue

            name, value_str, units = parts[0], parts[1], parts[2]
            extra = parts[3:]

            # Try converting the value to float
            try:
                value = float(value_str)
            except ValueError:
                value = value_str

            data[name] = {"value": value, "units": units, "extra": extra}

    return data


def get_value_SI(data, key):
    """
    Convert common Aviary units into SI units:
    - ft     -> meters
    - ft**2  -> m^2
    If unspecified, return the raw value.
    """
    entry = data[key]
    v = entry["value"]
    u = entry["units"]

    if u == "ft":
        return v * FT2M
    if u == "ft**2":
        return v * FT2M2
    return v


def build_vsp_from_aviary(csv_path,
                           vsp3_out="advanced_single_aisle.vsp3",
                           stl_out="advanced_single_aisle.stl"):
    # 1) Load CSV
    aviary = load_aviary_csv(csv_path)

    # 2) Reset VSP model
    if vsp is None:
        raise RuntimeError(
            "OpenVSP Python bindings not available. Install them or select the Python interpreter that has them."
        )

    vsp.VSPCheckSetup()
    vsp.VSPRenew()
    vsp.ClearVSPModel()
    vsp.DeleteAllResults()

     # -------------------------------------
    # FUSELAGE
    # -------------------------------------
    fus_len = get_value_SI(aviary, "aircraft:fuselage:length") 
    fus_h = get_value_SI(aviary, "aircraft:fuselage:max_height") 
    fus_w = get_value_SI(aviary, "aircraft:fuselage:max_width") 
    fus_d = max(fus_h, fus_w) # approximate diameter

    fus_id = vsp.AddGeom("FUSELAGE", "") 
    vsp.SetGeomName(fus_id, "Fuselage")

    # Set fuselage length 
    vsp.SetParmValUpdate(fus_id, "Length", "Design", fus_len)

    # Set nose/tail cross-section dimensions (approximate with RoundedRect) 
    xsec_surf = vsp.GetXSecSurf(fus_id, 0) 
    num_xsec = vsp.GetNumXSec(xsec_surf)

    for idx in [0, num_xsec - 1]:
        xsec_id = vsp.GetXSec(xsec_surf, idx)
        wid_id = vsp.GetXSecParm(xsec_id, "RoundedRect_Width")
        hei_id = vsp.GetXSecParm(xsec_id, "RoundedRect_Height")
        if vsp.ValidParm(wid_id): 
            vsp.SetParmVal(wid_id, fus_d) 
        if vsp.ValidParm(hei_id): 
            vsp.SetParmVal(hei_id, fus_d)

    # -------------------------------------
    # MAIN WING
    # -------------------------------------
    wing_span  = get_value_SI(aviary, "aircraft:wing:span")
    wing_area  = get_value_SI(aviary, "aircraft:wing:area")
    wing_taper = aviary["aircraft:wing:taper_ratio"]["value"]
    wing_sweep_deg = aviary["aircraft:wing:sweep"]["value"]  # already in degrees

    # Compute root/tip chord from area + span + taper
    wing_root_chord = 2.0 * wing_area / (wing_span * (1.0 + wing_taper))
    wing_tip_chord  = wing_root_chord * wing_taper

    wing_id = vsp.AddGeom("WING", "")
    vsp.SetGeomName(wing_id, "MainWing")

    # Total area/span
    vsp.SetParmValUpdate(wing_id, "TotalArea", "WingGeom", wing_area)
    vsp.SetParmValUpdate(wing_id, "TotalSpan", "WingGeom", wing_span)

    # Section parameters
    vsp.SetParmValUpdate(wing_id, "Sweep", "XSec_1", wing_sweep_deg)
    vsp.SetParmValUpdate(wing_id, "Root_Chord", "XSec_1", wing_root_chord)
    vsp.SetParmValUpdate(wing_id, "Tip_Chord",  "XSec_1", wing_tip_chord)

    # Position wing at 25% of fuselage length
    vsp.SetParmValUpdate(wing_id, "X_Rel_Location", "XForm", 0.40 * fus_len)

    #output wing parameters for CFD setup use:
    import csv
    wing_avg_chord = (wing_root_chord + wing_tip_chord) / 2.0

    with open("wing_output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wing_area", wing_area])
        writer.writerow(["wing_avg_chord", wing_avg_chord])
        writer.writerow(["fuselage_len", fus_len])

    # -------------------------------------
    # HORIZONTAL TAIL
    # -------------------------------------
    ht_area = get_value_SI(aviary, "aircraft:horizontal_tail:area")
    ht_AR   = aviary["aircraft:horizontal_tail:aspect_ratio"]["value"]
    ht_span = math.sqrt(ht_AR * ht_area)
    ht_taper = aviary["aircraft:horizontal_tail:taper_ratio"]["value"]
    ht_sweep_deg = aviary["aircraft:horizontal_tail:sweep"]["value"]

    ht_root_chord = 2.0 * ht_area / (ht_span * (1.0 + ht_taper))
    ht_tip_chord  = ht_root_chord * ht_taper

    ht_id = vsp.AddGeom("WING", wing_id)
    vsp.SetGeomName(ht_id, "HTail")

    vsp.SetParmValUpdate(ht_id, "TotalArea", "WingGeom", ht_area)
    vsp.SetParmValUpdate(ht_id, "TotalSpan", "WingGeom", ht_span)
    vsp.SetParmValUpdate(ht_id, "Sweep", "XSec_1", ht_sweep_deg)
    vsp.SetParmValUpdate(ht_id, "Root_Chord", "XSec_1", ht_root_chord)
    vsp.SetParmValUpdate(ht_id, "Tip_Chord",  "XSec_1", ht_tip_chord)

    # Place horizontal tail near aft fuselage
    vsp.SetParmValUpdate(ht_id, "X_Rel_Location", "XForm", fus_len - ht_root_chord * 1.2)
    vsp.SetParmValUpdate(ht_id, "Z_Rel_Location", "XForm", 0.0)

    # -------------------------------------
    # VERTICAL TAIL
    # -------------------------------------
    #'''
    vt_area = get_value_SI(aviary, "aircraft:vertical_tail:area")
    vt_AR   = aviary["aircraft:vertical_tail:aspect_ratio"]["value"]
    vt_span = math.sqrt(vt_AR * vt_area)
    vt_taper = aviary["aircraft:vertical_tail:taper_ratio"]["value"]
    vt_sweep_deg = aviary["aircraft:vertical_tail:sweep"]["value"]

    vt_root_chord = 2.0 * vt_area / (vt_span * (1.0 + vt_taper))
    vt_tip_chord  = vt_root_chord * vt_taper

    vt_id = vsp.AddGeom("WING", fus_id)
    vsp.SetGeomName(vt_id, "VTail")

    vsp.SetParmValUpdate(vt_id, "Sym_Planar_Flag", "Sym", 0)  #no symmetry
    vsp.SetParmValUpdate(vt_id, "TotalArea", "WingGeom", vt_area)
    vsp.SetParmValUpdate(vt_id, "TotalSpan", "WingGeom", vt_span)
    vsp.SetParmValUpdate(vt_id, "Sweep", "XSec_1", vt_sweep_deg)
    vsp.SetParmValUpdate(vt_id, "Root_Chord", "XSec_1", vt_root_chord)
    vsp.SetParmValUpdate(vt_id, "Tip_Chord",  "XSec_1", vt_tip_chord)

    # Rotate vertical tail upright and move aft
    vsp.SetParmValUpdate(vt_id, "X_Rel_Location", "XForm", fus_len - vt_root_chord *1.2)
    vsp.SetParmValUpdate(vt_id, "Z_Rel_Location", "XForm", fus_h * 0.1)
    vsp.SetParmValUpdate(vt_id, "X_Rel_Rotation", "XForm", 90.0)
    #'''
    
    # -------------------------------------
    # Finalize and Export
    # -------------------------------------
    vsp.Update()
    # Generate the mesh for all components
    #mesh_id = vsp.ComputeCompGeom(vsp.SET_ALL, False, vsp.NO_FILE_TYPE)

    vsp.WriteVSPFile(vsp3_out)
    
     # -------------------------------------
    # Finalize and Export
    # -------------------------------------
    vsp.Update()

    mesh_id = vsp.ComputeCompGeom(vsp.SET_ALL, False, vsp.NO_FILE_TYPE)
    
    vsp.WriteVSPFile(vsp3_out)

    # Export STL to the requested output path
    abs_stl = os.path.abspath(stl_out)

    # output STL file
    mesh_id = vsp.ExportFile(abs_stl, vsp.SET_ALL, vsp.EXPORT_STL)

#
if __name__ == "__main__":
    csv_path = "advanced_single_aisle_FLOPS.csv"  # Input Original CSV File again
    build_vsp_from_aviary(csv_path)
