import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Configuration de la plaque (20x20 cm)
plate_width = 20
plate_height = 20

# Position des capteurs : 4 coins + centre
sensor_positions = {
    "TL": (0, plate_height),       # Top Left
    "TR": (plate_width, plate_height), # Top Right
    "BL": (0, 0),                 # Bottom Left
    "BR": (plate_width, 0),       # Bottom Right
    "C": (plate_width / 2, plate_height / 2),  # Center
}

st.title("🛡️ Dashboard de simulation de blindage intelligent")

st.sidebar.header("⚙️ Simulation des signaux")
simulate = st.sidebar.button("🎯 Simuler un impact")

if "adc_values" not in st.session_state or simulate:
    # Simule des valeurs ADC réalistes (plus fort au centre, aléatoire autour)
    impact_x = np.random.uniform(0, plate_width)
    impact_y = np.random.uniform(0, plate_height)

    def simulate_adc(pos):
        # Distance à l'impact, plus c'est proche plus l'ADC est haut
        d = np.linalg.norm(np.array(pos) - np.array([impact_x, impact_y]))
        return max(0, 255 - d * 10 + np.random.normal(0, 5))  # bruit

    adc_values = {name: simulate_adc(pos) for name, pos in sensor_positions.items()}
    st.session_state.adc_values = adc_values
    st.session_state.impact = (impact_x, impact_y)

adc_values = st.session_state.adc_values

st.subheader("📟 Valeurs ADC des capteurs")
for name, val in adc_values.items():
    st.write(f"**{name}** ({sensor_positions[name]}): {val:.1f}")

# Interpolation sur toute la surface
points = np.array(list(sensor_positions.values()))
values = np.array([adc_values[k] for k in sensor_positions])

grid_x, grid_y = np.mgrid[0:plate_width:100j, 0:plate_height:100j]
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# Affichage du heatmap
fig, ax = plt.subplots(figsize=(6, 5))
c = ax.imshow(grid_z.T, extent=(0, plate_width, 0, plate_height), origin='lower', cmap='plasma')
ax.set_title("🗺️ Carte d'intensité de l'impact")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
fig.colorbar(c, ax=ax, label="Valeur ADC")

# Ajout des capteurs sur la carte
for name, (x, y) in sensor_positions.items():
    ax.plot(x, y, 'wo')
    ax.text(x + 0.5, y + 0.5, name, color='white')

# Position d'impact simulée
ix, iy = st.session_state.impact
ax.plot(ix, iy, 'rx', markersize=10, label="Impact réel")
ax.legend()

st.pyplot(fig)
