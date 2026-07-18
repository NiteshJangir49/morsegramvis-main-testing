import os
import ast
import numpy as np
import pandas as pd
import vtk

def generate_comparative_node_networks(particles_csv, contact_points_vtp, output_vtp_A, output_vtp_B, difference_ratio=0.05):
    """
    Extracts contact structures, finds the closest particle center pairs,
    calculates physical load vectors, and pools those weights directly onto the
    particle NODES (points) instead of edges. Exports two node-scalar VTP datasets.
    """
    print("[Node Engine] Parsing particle datasets...")
    df_part = pd.read_csv(particles_csv)
    
    # 1. EXTRACT BEAD CENTERS FROM STATISTICS CSV
    points = vtk.vtkPoints()
    center_coords = []
    id_to_idx = {}
    point_counter = 0
    
    print("[Node Engine] Gathering 3D particle coordinate nodes...")
    for _, row in df_part.iterrows():
        cp_id = int(row['cp_id'])
        try:
            centroid = ast.literal_eval(row['centroid'])
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
            pos = (cx, cy, cz)
            center_coords.append(pos)
            points.InsertNextPoint(cx, cy, cz)
            id_to_idx[cp_id] = point_counter
            point_counter += 1
        except Exception:
            continue

    num_nodes = points.GetNumberOfPoints()
    print(f"[Node Engine] Successfully mapped {num_nodes} particle nodes.")

    # 2. EXTRACT CONTACT INTERSECTION POINTS FROM VTP FILE
    print(f"[Node Engine] Extracting contact intersections from: {os.path.basename(contact_points_vtp)}")
    fresh_reader = vtk.vtkXMLPolyDataReader()
    fresh_reader.SetFileName(contact_points_vtp)
    fresh_reader.Update()
    
    contact_poly = fresh_reader.GetOutput()
    if not contact_poly or not contact_poly.GetPoints():
        print("[Node Engine Error] Target contact VTP file contains no geometric points.")
        return

    contact_points_array = contact_poly.GetPoints()
    num_contact_pts = contact_points_array.GetNumberOfPoints()
    print(f"[Node Engine] Found {num_contact_pts} contact intersection regions.")

    # 3. SPATIAL PROXIMITY PAIRING (Find interacting nodes)
    edge_list = []
    
    print("[Node Engine] Resolving geometric contact pairings...")
    for i in range(num_contact_pts):
        tx, ty, tz = contact_points_array.GetPoint(i)
        
        # Distance squared calculation loop to find the closest 2 particle centers
        dists = [((tx - cx)**2 + (ty - cy)**2 + (tz - cz)**2, idx) for idx, (cx, cy, cz) in enumerate(center_coords)]
        dists.sort(key=lambda x: x[0])
        
        if len(dists) >= 2:
            center_idx_a = dists[0][1]
            center_idx_b = dists[1][1]
            
            if center_idx_a != center_idx_b:
                # Store the interacting pair indices
                edge_list.append((center_idx_a, center_idx_b))

    # 4. GENERATE FORCE COEFFS AND POOL THEM DIRECTLY ONTO NODES
    np.random.seed(42) # Safe global seed for cross-run stability
    num_contacts = len(edge_list)
    
    # Determine which contact interactions undergo mutation (5%)
    different_mask = np.random.rand(num_contacts) < difference_ratio if num_contacts > 0 else np.array([], dtype=bool)

    # Initialize accumulating load arrays for the particle nodes
    node_load_A = np.zeros(num_nodes)
    node_load_B = np.zeros(num_nodes)

    mutated_contact_count = 0
    for i, (idx_a, idx_b) in enumerate(edge_list):
        p1 = np.array(center_coords[idx_a])
        p2 = np.array(center_coords[idx_b])
        distance = np.linalg.norm(p1 - p2)
        if distance == 0: distance = 1e-4

        # Calculate a realistic physics baseline load vector
        base_weight = 75.0 / distance

        if not different_mask[i]:
            # 95% of interactions: Both participating particles receive identical load changes
            shared_noise = np.random.uniform(0.90, 1.10)
            weight_val = base_weight * shared_noise
            
            node_load_A[idx_a] += weight_val
            node_load_A[idx_b] += weight_val
            
            node_load_B[idx_a] += weight_val
            node_load_B[idx_b] += weight_val
        else:
            # 5% of interactions: Particle loads branch off independently
            weight_A = base_weight * np.random.uniform(0.85, 1.15)
            weight_B = base_weight * np.random.uniform(0.85, 1.15)
            
            node_load_A[idx_a] += weight_A
            node_load_A[idx_b] += weight_A
            
            node_load_B[idx_a] += weight_B
            node_load_B[idx_b] += weight_B
            mutated_contact_count += 1

    print(f"[Node Engine] Accumulated force loads. Distributed {mutated_contact_count} / {num_contacts} localized mutations.")

    # 5. EXPORT VOLUME A AS A NODE CLOUD
    print(f"[Node Engine] Writing Node Volume A -> {os.path.basename(output_vtp_A)}")
    scalars_A = vtk.vtkFloatArray()
    scalars_A.SetName("NodeWeight")
    for val in node_load_A:
        scalars_A.InsertNextValue(float(val))

    poly_A = vtk.vtkPolyData()
    poly_A.SetPoints(points)
    poly_A.GetPointData().SetScalars(scalars_A) # Bind directly as PointData scalars
    
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(poly_A)
    writer.SetDataModeToBinary()
    writer.SetFileName(output_vtp_A)
    writer.Write()

    # 6. EXPORT VOLUME B AS A NODE CLOUD
    print(f"[Node Engine] Writing Node Volume B -> {os.path.basename(output_vtp_B)}")
    scalars_B = vtk.vtkFloatArray()
    scalars_B.SetName("NodeWeight")
    for val in node_load_B:
        scalars_B.InsertNextValue(float(val))

    poly_B = vtk.vtkPolyData()
    poly_B.SetPoints(points)
    poly_B.GetPointData().SetScalars(scalars_B) # Bind directly as PointData scalars
    
    writer.SetFileName(output_vtp_B)
    writer.SetInputData(poly_B)
    writer.Write()
    
    print("[Node Engine] Comparative node VTP generation finished successfully.")

# Configuration data mapping
if __name__ == "__main__":
    base_dir = r"C:/Users/nites/Downloads/morsegramvis-main/morsegramvis-main"
    
    particles_file = os.path.join(base_dir, "dataset_files", "Steel Beads", "data", "all_particle_stats.csv")
    contact_vtp_source = os.path.join(base_dir, "dataset_files", "Steel Beads", "chamf_distance_Steel_Deposition_181_176_251_downsampled_contacts.vtp")
    
    # New separate export naming scheme target locations
    vtp_A_nodes_out = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_A.vtp")
    vtp_B_nodes_out = os.path.join(base_dir, "dataset_files", "Steel Beads", "visualization", "node_network_volume_B.vtp")
    
    os.makedirs(os.path.dirname(vtp_A_nodes_out), exist_ok=True)
    
    generate_comparative_node_networks(
        particles_file, contact_vtp_source, vtp_A_nodes_out, vtp_B_nodes_out, difference_ratio=0.05
    )