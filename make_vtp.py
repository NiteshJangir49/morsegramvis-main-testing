import os
import ast
import numpy as np
import pandas as pd
import vtk
from tqdm import tqdm

def generate_comparative_geocentric_networks(particles_csv, contact_points_vtp, output_vtp_A, output_vtp_B, difference_ratio=0.05):
    """
    Extracts contact positions from a VTP file, finds the closest particle centers 
    geometrically, generates realistic comparative weights, and exports two VTP networks.
    Only a small fraction (default 5%) of edges will contain different weights.
    """
    print("[Network Engine] Parsing particle datasets...")
    df_part = pd.read_csv(particles_csv)
    
    # 1. EXTRACT BEAD CENTERS FROM STATISTICS CSV
    points = vtk.vtkPoints()
    center_coords = []
    id_to_idx = {}
    point_counter = 0
    
    print("[Network Engine] Gathering 3D physical coordinates...")
    for _, row in tqdm(df_part.iterrows(), total=len(df_part)):
        cp_id = int(row['cp_id'])
        try:
            centroid = ast.literal_eval(row['centroid'])
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
            pos = (cx, cy, cz)
            center_coords.append(pos)
            vtk_points_inserted = points.InsertNextPoint(cx, cy, cz)
            id_to_idx[cp_id] = point_counter
            point_counter += 1
        except Exception:
            continue

    num_centers = points.GetNumberOfPoints()
    print(f"[Network Engine] Successfully mapped {num_centers} center nodes.")

    # 2. EXTRACT CONTACT INTERSECTION POINTS FROM VTP FILE
    print(f"[Network Engine] Extracting contact coordinates from: {os.path.basename(contact_points_vtp)}")
    fresh_reader = vtk.vtkXMLPolyDataReader()
    fresh_reader.SetFileName(contact_points_vtp)
    fresh_reader.Update()
    
    contact_poly = fresh_reader.GetOutput()
    if not contact_poly or not contact_poly.GetPoints():
        print("[Network Engine Error] Target contact VTP file contains no geometric points.")
        return

    contact_points_array = contact_poly.GetPoints()
    num_contact_pts = contact_points_array.GetNumberOfPoints()
    print(f"[Network Engine] Found {num_contact_pts} contact intersection regions.")

    # 3. SPATIAL PROXIMITY PAIRING (Geometric Lookup Loop)
    seen_edges = set()
    edge_list = []
    
    print("[Network Engine] Calculating spatial proximity linkages...")
    for i in tqdm(range(num_contact_pts), total=num_contact_pts):
        tx, ty, tz = contact_points_array.GetPoint(i)
        
        # Distance squared calculation loop to find the closest 2 particle centers
        dists = []
        for idx, (cx, cy, cz) in enumerate(center_coords):
            d = (tx - cx)**2 + (ty - cy)**2 + (tz - cz)**2
            dists.append((d, idx))
        
        dists.sort(key=lambda x: x[0])
        
        if len(dists) >= 2:
            center_idx_a = dists[0][1]
            center_idx_b = dists[1][1]
            
            if center_idx_a != center_idx_b:
                edge_pair = tuple(sorted((center_idx_a, center_idx_b)))
                if edge_pair not in seen_edges:
                    seen_edges.add(edge_pair)
                    edge_list.append(edge_pair)

    num_edges = len(edge_list)
    print(f"[Network Engine] Built {num_edges} unique contact lines. Generating dual scalar weights...")

    # 4. GENERATE COMPARATIVE WEIGHT MATRICES (MODIFIED FOR 5% DIFFERENCE PROFILE)
    np.random.seed(42) # Safe global seed for cross-run stability
    
    # Randomly target roughly 5% of edges to follow distinct delta profiles
    different_mask = np.random.rand(num_edges) < difference_ratio if num_edges > 0 else np.array([], dtype=bool)

    weights_A = np.zeros(num_edges)
    weights_B = np.zeros(num_edges)

    mutated_count = 0
    for i, edge in enumerate(edge_list):
        p1 = np.array(center_coords[edge[0]])
        p2 = np.array(center_coords[edge[1]])
        distance = np.linalg.norm(p1 - p2)
        if distance == 0: distance = 1e-4

        # Calculate a realistic physics baseline load vector
        base_weight = 75.0 / distance

        if not different_mask[i]:
            # ~95% of Edges: Equal Weights across both volumes
            shared_noise = np.random.uniform(0.90, 1.10)
            weights_A[i] = base_weight * shared_noise
            weights_B[i] = base_weight * shared_noise
        else:
            # ~5% of Edges: Unique weights branching independently
            noise_A = np.random.uniform(0.85, 1.15)
            noise_B = np.random.uniform(0.85, 1.15)
            weights_A[i] = base_weight * noise_A
            weights_B[i] = base_weight * noise_B
            mutated_count += 1

    print(f"[Network Engine] Configured {mutated_count} / {num_edges} variations (~{mutated_count/num_edges*100:.2f}%)")

    # 5. EXPORT VOLUME A
    print(f"[Network Engine] Writing Volume A -> {os.path.basename(output_vtp_A)}")
    lines_A = vtk.vtkCellArray()
    scalars_A = vtk.vtkFloatArray()
    scalars_A.SetName("ContactWeight")
    
    for i, edge in enumerate(edge_list):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge[0])
        line.GetPointIds().SetId(1, edge[1])
        lines_A.InsertNextCell(line)
        scalars_A.InsertNextValue(float(weights_A[i]))

    poly_A = vtk.vtkPolyData()
    poly_A.SetPoints(points)
    poly_A.SetLines(lines_A)
    poly_A.GetCellData().SetScalars(scalars_A)
    
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(poly_A)
    writer.SetDataModeToBinary()
    writer.SetFileName(output_vtp_A)
    writer.Write()

    # 6. EXPORT VOLUME B
    print(f"[Network Engine] Writing Volume B -> {os.path.basename(output_vtp_B)}")
    lines_B = vtk.vtkCellArray()
    scalars_B = vtk.vtkFloatArray()
    scalars_B.SetName("ContactWeight")
    
    for i, edge in enumerate(edge_list):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge[0])
        line.GetPointIds().SetId(1, edge[1])
        lines_B.InsertNextCell(line)
        scalars_B.InsertNextValue(float(weights_B[i]))

    poly_B = vtk.vtkPolyData()
    poly_B.SetPoints(points)
    poly_B.SetLines(lines_B)
    poly_B.GetCellData().SetScalars(scalars_B)
    
    writer.SetFileName(output_vtp_B)
    writer.SetInputData(poly_B)
    writer.Write()
    
    print("[Network Engine] Comparative VTP generation pipeline finished successfully.")

# Configuration data mapping
if __name__ == "__main__":
    base_dir = r"C:/Users/nites/Downloads/morsegramvis-main/morsegramvis-main"
    
    particles_file = os.path.join(base_dir, "dataset_files", "Steel Beads", "data", "all_particle_stats.csv")
    contact_vtp_source = os.path.join(base_dir, "dataset_files", "Steel Beads", "chamf_distance_Steel_Deposition_181_176_251_downsampled_contacts.vtp")
    
    vtp_A_out = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "contact_network_volume_A.vtp")
    vtp_B_out = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "contact_network_volume_B.vtp")
    
    os.makedirs(os.path.dirname(vtp_A_out), exist_ok=True)
    
    # Run the generator function targeting a clear 5% delta difference output
    generate_comparative_geocentric_networks(
        particles_file, contact_vtp_source, vtp_A_out, vtp_B_out, difference_ratio=0.05
    )