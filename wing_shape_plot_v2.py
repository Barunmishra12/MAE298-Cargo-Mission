import numpy as np
import matplotlib.pyplot as plt

# =======================================================
# Function to compute and return wing planform coordinates
# =======================================================

def compute_wing_planform(AR, S, b, c_mean, sweep_deg, color="k", label="Wing"):
    """
    Returns x and y coordinates for a straight-tapered wing planform.
    """

    sweep = np.radians(sweep_deg)

    # Initial estimate for root chord
    c_root = (2 * S) / (b * (1 + 0.4))  # assume taper ~ 0.4 initially
    
    # Solve for taper ratio using MAC relation
    try:
        import mpmath as mp
        f = lambda lam: (2/3) * c_root * (1 + lam + lam**2) / (1 + lam) - c_mean
        taper = float(mp.findroot(f, 0.5))
    except:
        taper = 0.4  # fallback
    
    # Tip chord
    c_tip = taper * c_root
    half_span = b / 2
    x_le_tip = half_span * np.tan(sweep)

    # Coordinates (closed loop)
    x = [
        0,                       # root leading edge
        c_root,                  # root trailing edge
        x_le_tip + c_tip,        # tip trailing edge
        x_le_tip,                # tip leading edge
        0                        # close shape
    ]
    y = [
        0,
        0,
        half_span,
        half_span,
        0
    ]

    return np.array(x), np.array(y), color, label


# =======================================================
# Define multiple wing versions here
# =======================================================

wing_configs = [
    # AR,  S,  b,  MAC, sweep, color, label
    (11.558, 1220.0, 118.75, 10.75, 23.6, "b", "Baseline"),
    (20, 1600, 178.9, 8.94, 30, "r", "Optimized"),
]


# =======================================================
# Plot all wings
# =======================================================

plt.figure(figsize=(12, 5))

for cfg in wing_configs:
    AR, S, b, c_mean, sweep, color, label = cfg
    x, y, color, label = compute_wing_planform(AR, S, b, c_mean, sweep, color, label)
    plt.plot(x, y, color=color, linewidth=2, label=label)
    plt.fill(x, y, alpha=0.25, color=color)

plt.title("Baseline vs Optimized Wing")
plt.xlabel("Chordwise Distance (m)")
plt.ylabel("Spanwise Distance (m)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
