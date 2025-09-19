import streamlit as st
import trimesh
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, Tuple

# ================================
# CONFIGURATION & DATA
# ================================
st.set_page_config(page_title="3D Model & Cost Analyzer", layout="wide")

USD_TO_INR_RATE = 83.50

MATERIALS = {
    "ABS": {"density": 1.04, "cost_usd_per_kg": 30},
    "PC-ABS": {"density": 1.15, "cost_usd_per_kg": 45},
    "Nylon (PA66)": {"density": 1.14, "cost_usd_per_kg": 60},
    "HDPE": {"density": 0.95, "cost_usd_per_kg": 20},
    "Aluminum (6061)": {"density": 2.70, "cost_usd_per_kg": 15},
    "Stainless Steel (316L)": {"density": 8.00, "cost_usd_per_kg": 25},
}

MOLD_COST_FACTORS = {
    "base_cost_usd": 1500, "cost_per_cm3_usd": 10,
    "complexity_multiplier": {"Low": 1.0, "Medium": 1.8, "High": 3.5},
    "cavity_multiplier": {1: 1.0, 2: 1.5, 4: 2.2, 8: 3.0},
    "processing_cost_per_hour_usd": 50, "cycle_time_base_seconds": 20,
    "cycle_time_per_cm3_seconds": 0.2,
}

FILE_EXTENSIONS = {"STL": ["stl"], "STEP": ["step", "stp"], "Parasolid": ["x_t", "x_b"], "NX Part": ["prt"]}

# ================================
# CORE CALCULATION FUNCTIONS
# ================================

def analyze_mesh_properties(mesh: trimesh.Trimesh) -> Tuple[Dict[str, Any], float, np.ndarray]:
    """Analyzes mesh for geometric properties."""
    if mesh.is_empty: return {}, 0.0, np.array([0, 0, 0])
    
    bounds, surface_area = mesh.bounding_box.extents, mesh.area / 100.0
    volume_cm3 = (mesh.volume if mesh.is_watertight else mesh.convex_hull.volume) / 1000.0
    if not mesh.is_watertight:
        st.warning("⚠️ Warning: Mesh is not watertight. Volume based on convex hull.")
    
    center_of_mass_cm = mesh.center_mass / 10.0
    model_info = {
        "Bounding Box (cm)": f"W: {bounds[0]/10:.2f}, D: {bounds[1]/10:.2f}, H: {bounds[2]/10:.2f}",
        "Volume (cm³)": f"{volume_cm3:.3f}",
        "Is Watertight?": "✅ Yes" if mesh.is_watertight else "❌ No",
        "Faces (Triangles)": f"{len(mesh.faces):,}",
    }
    return model_info, volume_cm3, center_of_mass_cm

def calculate_initial_costs(volume_cm3: float, complexity: str, cavities: int) -> Tuple[pd.DataFrame, float]:
    """Calculates initial part and mold costs based on automated estimates."""
    part_cost_rows = []
    for mat, props in MATERIALS.items():
        mass_g = volume_cm3 * props["density"]
        material_cost_usd = (mass_g / 1000.0) * props["cost_usd_per_kg"]
        cycle_time_sec = MOLD_COST_FACTORS["cycle_time_base_seconds"] + (volume_cm3 * MOLD_COST_FACTORS["cycle_time_per_cm3_seconds"])
        processing_cost_usd = (MOLD_COST_FACTORS["processing_cost_per_hour_usd"] / 3600) * cycle_time_sec
        total_part_cost_usd = material_cost_usd + processing_cost_usd
        total_part_cost_inr = total_part_cost_usd * USD_TO_INR_RATE
        part_cost_rows.append([mat, f"{mass_g:.2f}", total_part_cost_usd, total_part_cost_inr])

    part_cost_df = pd.DataFrame(part_cost_rows, columns=["Material", "Weight (g)", "Part Cost (USD)", "Part Cost (INR)"])
    
    base_mold_cost = MOLD_COST_FACTORS["base_cost_usd"] + (volume_cm3 * MOLD_COST_FACTORS["cost_per_cm3_usd"])
    mold_cost_usd = base_mold_cost * MOLD_COST_FACTORS["complexity_multiplier"][complexity] * MOLD_COST_FACTORS["cavity_multiplier"][cavities]
    
    return part_cost_df, mold_cost_usd

def calculate_amortized_costs(part_cost_df: pd.DataFrame, mold_cost_usd: float, amortization_material: str) -> pd.DataFrame:
    """Calculates the final cost per part after factoring in mold cost over production volumes."""
    production_volumes = [1000, 5000, 25000, 100000, 500000]
    
    # Extract the specific part cost for the chosen material from the DataFrame
    try:
        material_row = part_cost_df[part_cost_df["Material"] == amortization_material]
        if material_row.empty:
            return pd.DataFrame()
        example_part_cost_usd = float(material_row.iloc[0]["Part Cost (USD)"])
    except (ValueError, IndexError):
        # Handle cases where the cost is not a valid number or the material isn't found
        return pd.DataFrame()

    amortized_rows = []
    for vol in production_volumes:
        amortized_mold_cost_per_part = mold_cost_usd / vol
        total_cost_per_part_usd = example_part_cost_usd + amortized_mold_cost_per_part
        total_cost_per_part_inr = total_cost_per_part_usd * USD_TO_INR_RATE
        amortized_rows.append([f"{vol:,}", f"${total_cost_per_part_usd:.3f}", f"₹{total_cost_per_part_inr:.2f}"])

    return pd.DataFrame(amortized_rows, columns=["Production Volume", "Amortized Cost/Part (USD)", "Amortized Cost/Part (INR)"])

# ================================
# UI & MAIN LOGIC
# ================================
def main():
    st.title("🛠️ Advanced 3D Model & Manufacturing Cost Analyzer")

    # --- Sidebar Inputs ---
    st.sidebar.header("1. Model & Parameters")
    file_type = st.sidebar.selectbox("File Type", list(FILE_EXTENSIONS.keys()))
    uploaded_file = st.sidebar.file_uploader("Upload 3D Model", type=FILE_EXTENSIONS[file_type])
    part_complexity = st.sidebar.select_slider("Part Complexity", options=["Low", "Medium", "High"], value="Medium")
    num_cavities = st.sidebar.select_slider("Mold Cavities", options=[1, 2, 4, 8], value=1)
    
    st.sidebar.header("2. Cost Calculation Method")
    cost_input_method = st.sidebar.radio("Per-Part Cost", ["Automated Estimation", "Manual Input"])

    # --- Main Panel ---
    if not uploaded_file:
        st.info("Upload a 3D model and set parameters in the sidebar to begin analysis.")
        return

    try:
        with st.spinner("Analyzing model..."):
            file_ext = uploaded_file.name.split('.')[-1].lower()
            mesh = trimesh.load(uploaded_file, file_type=file_ext, force="mesh")
        
        model_info, volume, com = analyze_mesh_properties(mesh)
        if not model_info:
            st.error("❌ Model is empty or could not be processed.")
            return

        # Use session state to store and manage cost data across reruns
        if 'part_cost_df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            st.session_state.part_cost_df, st.session_state.mold_cost_usd = calculate_initial_costs(volume, part_complexity, num_cavities)
            st.session_state.file_name = uploaded_file.name

        tab1, tab2, tab3 = st.tabs(["💰 Cost Analysis", "📊 Model Properties", "🎨 3D Visualization"])

        with tab2:
            st.subheader("Key Geometric Properties")
            st.table(pd.DataFrame(model_info.items(), columns=["Property", "Value"]))

        with tab3:
            st.plotly_chart(create_3d_plot(mesh, com), use_container_width=True)

        with tab1:
            st.subheader("⚙️ Manufacturing Cost Estimation")
            st.markdown(f"**Assumptions:** Part Complexity: `{part_complexity}`, Mold Cavities: `{num_cavities}`")
            
            # --- Part Cost Section ---
            st.markdown("#### Per-Part Cost (Material + Processing)")
            if cost_input_method == "Manual Input":
                st.info("💡 You are in manual mode. Edit the cost values in the table below.")
                # The data editor allows direct user modification
                edited_df = st.data_editor(
                    st.session_state.part_cost_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Part Cost (USD)": st.column_config.NumberColumn(format="$%.3f"),
                        "Part Cost (INR)": st.column_config.NumberColumn(format="₹%.2f"),
                    }
                )
                # Save the user's edits back to the session state
                st.session_state.part_cost_df = edited_df
            else:
                st.dataframe(st.session_state.part_cost_df, use_container_width=True, hide_index=True)

            # --- Mold & Amortized Cost Section ---
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Tooling (Mold) Cost")
                mold_cost_inr = st.session_state.mold_cost_usd * USD_TO_INR_RATE
                st.metric("Estimated Mold Cost (USD)", f"${st.session_state.mold_cost_usd:,.2f}")
                st.metric("Estimated Mold Cost (INR)", f"₹{mold_cost_inr:,.2f}")

            with col2:
                st.markdown("#### Amortized Cost Analysis")
                amortization_material = st.selectbox(
                    "Select Material for Amortization",
                    options=st.session_state.part_cost_df["Material"].tolist()
                )
                amortized_df = calculate_amortized_costs(st.session_state.part_cost_df, st.session_state.mold_cost_usd, amortization_material)
                st.dataframe(amortized_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def create_3d_plot(mesh: trimesh.Trimesh, center_of_mass_cm: np.ndarray) -> go.Figure:
    """Generates an interactive 3D plot of the mesh."""
    vertices, faces = mesh.vertices, mesh.faces
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    fig = go.Figure(data=[
        go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='deepskyblue', opacity=0.7, name='Model'),
        go.Scatter3d(x=[center_of_mass_cm[0]*10], y=[center_of_mass_cm[1]*10], z=[center_of_mass_cm[2]*10], mode='markers', marker=dict(size=10, color='red', symbol='cross'), name='Center of Mass')
    ])
    fig.update_layout(scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)', aspectmode='data'), margin=dict(l=10, r=10, t=10, b=10), height=500)
    return fig

if __name__ == "__main__":
    main()
