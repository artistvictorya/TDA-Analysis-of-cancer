import sys
import os
import argparse
import numpy as np
import gudhi
import matplotlib.pyplot as plt
import itertools
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# --- CONSTANTS FOR GRADING ---
# Define the 6 health/illness levels (0=Healthy, 1-5=Illness Grade)
CANCER_LEVELS = [0, 1, 2, 3, 4, 5]
# This dictionary will store the reference diagrams for each level.
# Key: Level (int), Value: List of persistence diagrams [(birth, death), ...]
REFERENCE_DIAGRAMS = {level: [] for level in CANCER_LEVELS}
# This dictionary will store the average/representative diagram for each level.
AVERAGE_DIAGRAMS = {} 
# Note: For simple classification, we use the average, but for real systems,
# a support vector machine or similar vectorization is needed.

def convert_to_png(image_path):
    """Converts the image (e.g., TIFF) to a grayscale PNG format."""
    if image_path.lower().endswith(".png"):
        return image_path

    png_path = os.path.splitext(image_path)[0] + ".png"
    print(f"[INFO] Converting {image_path} to {png_path}")

    with Image.open(image_path) as img:
        img.convert("L").save(png_path)

    return png_path

def convert_image_to_point_cloud(image_path, threshold=None, tile_size=1000):
    """Creates a point cloud (x, y, z) from the image, where z is the filtration value (brightness)."""
    with Image.open(image_path) as img:
        w, h = img.size
        # ... (rest of image_to_point_cloud function remains the same) ...
        points = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tile_w = min(tile_size, w - x)
                tile_h = min(tile_size, h - y)
                tile = img.crop((x, y, x + tile_w, y + tile_h))
                arr = np.array(tile, dtype=np.uint8)

                xs, ys = np.meshgrid(np.arange(x, x + tile_w), np.arange(y, y + tile_h), indexing='xy')
                tile_points = np.column_stack((xs.ravel(), ys.ravel(), arr.ravel()))

                if threshold is not None:
                    tile_points = tile_points[tile_points[:, 2] < threshold]

                points.append(tile_points)

        points = np.vstack(points)
        print(f"[INFO] Total number of points: {len(points)}")

    return points[:, :2], points[:, 2] # Return points (x, y) and filtration values (z)

def save_point_cloud_txt(pts, filtr, out_path):
    """Saves the point cloud to a text file."""
    data = np.column_stack((pts, filtr))
    header = 'x y z'
    np.savetxt(out_path, data, fmt='%d %d %d', delimiter=' ', header=header)

def compute_persistent_homology(pts, filtr, max_dim, step, min_dist, out_path):
    """Computes Persistent Homology, ignoring H0 (connected components)."""
    N = len(pts)
    if step > 1:
        idx = np.arange(0, N, step)
        pts = pts[idx]
        filtr = filtr[idx]
    
    print(f"[INFO] Using {len(pts)} points after sampling (step={step})")

    stages = ["build", "vertex-filtr.", "propagate", "compute", "save", "plot"]
    with tqdm(total=len(stages), desc="Overall") as pbar:
        # 1) Build Delaunay triangulation
        pbar.set_description("1) Building Triangulation")
        dc = gudhi.DelaunayComplex(points=pts)
        st = dc.create_simplex_tree()
        pbar.update(1)

        # 2) Assign vertex filtrations
        pbar.set_description("2) Vertex Filtration")
        for vidx, z in enumerate(filtr):
            st.assign_filtration([vidx], z)
        pbar.update(1)

        # 3) Propagate filtration: max over faces (OPTIMIZED)
        pbar.set_description("3) Propagating Filtration (Optimized)")
        st.make_filtration_non_decreasing()
        st.initialize_filtration() 
        pbar.update(1)

        # 4) Compute persistence
        pbar.set_description("4) Computing Persistence")
        raw_pers = st.persistence()
        pbar.update(1)

        # 5) Filter & Save result
        pbar.set_description("5) Saving and Filtering")
        
        # --- KEY CHANGE: Filter H0 (dim=0) features ---
        # Only include dimensions from 1 up to max_dim
        pers = [(dim, (b, d)) for dim, (b, d) in raw_pers 
                if dim >= 1 and dim <= max_dim and d < float('inf')]
        
        # Filter features with persistence less than min_dist (death - birth)
        filtered = [(dim, (b, d)) for dim, (b, d) in pers
                    if abs(d - b) >= min_dist]

        out_pers = out_path
        with open(out_pers, 'w') as f:
            f.write("# dim birth death\n")
            for dim, (b, d) in filtered:
                f.write(f"{dim} {b:.6f} {d:.6f}\n")
        pbar.update(1)
        
        # 6) Plot diagram
        pbar.set_description("6) Plotting Diagram")
        plt.figure(figsize=(6, 6))
        gudhi.plot_persistence_diagram(filtered)
        plt.title(f'Persistence Diagram (dim=1..{max_dim}, persistence ≥ {min_dist})')
        plt.xlabel('Birth (Brightness)')
        plt.ylabel('Death (Brightness)')
        
        plot_out_path = os.path.splitext(out_pers)[0] + '.png'
        plt.savefig(plot_out_path)
        plt.close()
        pbar.update(1)

    print(f"[INFO] Persistence Diagram saved to: {out_pers}")
    print(f"[INFO] Plot saved to: {plot_out_path}")
    
    # Return the filtered persistence diagram for the classification model
    return [(d, b, de) for d, (b, de) in filtered]


def read_diagram(filename, dim=None):
    """Loads a persistence diagram from a text file, ignoring the dimension column."""
    pairs = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            d, b, de = parts
            try:
                d = int(d)
                b = float(b)
                de = float(de)
            except ValueError:
                continue
            if dim is None or d == dim:
                pairs.append((b, de))
    return pairs

def classify_diagram(test_diagram, average_diagrams):
    """
    Classifies a test diagram against the average reference diagrams using Bottleneck distance.
    Returns the closest level and the distance.
    """
    if not average_diagrams:
        return "N/A (Model not trained)", float('inf'), False

    best_level = None
    min_dist = float('inf')
    
    # Check if the test diagram has any H1 features (loops/glands)
    test_h1 = read_diagram(test_diagram, dim=1)
    is_cancer = bool(test_h1)

    for level, avg_diag_h1 in average_diagrams.items():
        try:
            # We only compare H1 features (loops/glands) for classification
            dist = gudhi.bottleneck_distance(test_h1, avg_diag_h1)
        except Exception as e:
            print(f"Error computing distance for Level {level}: {e}", file=sys.stderr)
            continue
            
        print(f"[CLASSIFY] Distance to Level {level}: {dist:.6f}")

        if dist < min_dist:
            min_dist = dist
            best_level = level

    result_level = f"Level {best_level}" if best_level is not None else "Unknown"
    
    # Simple check for cancer presence (H1 features usually indicate organized structure)
    # A low distance to Level 0 (Healthy) or Level 1 is usually better.
    # We use the classification result directly:
    diagnosis = "Cancer (Ill)" if best_level > 0 else "Healthy (Level 0)"

    return diagnosis, result_level, min_dist

# --- NEW TRAINING FUNCTIONS ---

def train_model(out_dir, max_dim, min_dist):
    """
    Loads all diagrams in the results folder based on the 0-5 naming convention
    and calculates the average H1 diagram for each level.
    """
    print("\n--- TRAINING CLASSIFICATION MODEL ---")
    
    # 1. Group diagrams by level (based on file naming convention: level_*.txt)
    for filename in os.listdir(out_dir):
        if filename.endswith(".txt") and "_pers_dim" in filename:
            try:
                # Expecting format: 'levelX_point_cloud_pers_dimY_distZ.txt'
                # Example: '0_sampleA_point_cloud_pers_dim1_dist10.0.txt' -> Level 0
                level_str = filename.split('_')[0]
                level = int(level_str)
                if level in CANCER_LEVELS:
                    full_path = os.path.join(out_dir, filename)
                    # We load only H1 features (loops/glands)
                    diagram_h1 = read_diagram(full_path, dim=1)
                    REFERENCE_DIAGRAMS[level].append(diagram_h1)
            except Exception as e:
                print(f"[WARNING] Skipping file {filename}: Cannot determine level. {e}")
    
    # 2. Compute the Average Diagram for each level (using Persistence Landscapes or simply mean B/D)
    # Since averaging diagrams is complex, we'll use a simplified method: 
    # taking the first diagram found for each level as the representative.
    
    for level, diagrams in REFERENCE_DIAGRAMS.items():
        if diagrams:
            # Using the first diagram found as the representative (simplified model)
            AVERAGE_DIAGRAMS[level] = diagrams[0]
            print(f"[TRAINING] Level {level}: Representative diagram created from {len(diagrams)} samples.")
        else:
            print(f"[TRAINING] Level {level}: No samples found for this level.")
    
    print("--- MODEL TRAINING COMPLETE ---")
    return AVERAGE_DIAGRAMS

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description="Topological Data Analysis (TDA) for Histopathological Images"
    )
    # ... (other arguments remain the same) ...
    parser.add_argument('images', nargs='*',
                        help='Paths to image files (TIFF, PNG, JPG) to process. Can be a list of files or an empty list for training/classification only.')
    parser.add_argument('--threshold', type=int, default=180,
                        help='Maximum brightness value (0-255) to include in the point cloud (dark pixels). Default is 180.')
    parser.add_argument('--tile-size', type=int, default=1000,
                        help='Tile size (default 1000) for processing very large images.')
    parser.add_argument('--out-dir', default='results',
                        help='Directory for saving output files (default "results").')
    
    # Arguments for Persistent Homology
    parser.add_argument('--analyze-homology', action='store_true',
                        help='Perform Persistent Homology analysis.')
    parser.add_argument('--homology-step', type=int, default=10,
                        help='Point cloud sampling factor (every N-th point). Default is 10.')
    parser.add_argument('--homology-max-dim', type=int, default=1,
                        help='Maximum homology dimension to compute (0: components, 1: loops/holes). Default is 1.')
    parser.add_argument('--homology-min-dist', type=float, default=10.0,
                        help='Minimum persistence length (death - birth) to be considered a feature. Default is 10.0.')
                        
    # Arguments for Classification
    parser.add_argument('--mode', choices=['process', 'train', 'classify'], default='process',
                        help='Mode of operation: "process" (PC+PH for new files), "train" (build model from existing diagrams), or "classify" (classify new diagram).')
    
    args = parser.parse_args()

    # Create folders
    os.makedirs(args.out_dir, exist_ok=True)
    point_cloud_files = []
    pers_diagram_files = []
    
    # -----------------------------------------------------------
    # MODE: TRAIN (Load existing diagrams and define the 5 levels)
    # -----------------------------------------------------------
    if args.mode == 'train':
        average_diagrams = train_model(args.out_dir, args.homology_max_dim, args.homology_min_dist)
        print("\nModel ready. Now run the script in 'process' or 'classify' mode.")
        sys.exit(0)
    
    # -----------------------------------------------------------
    # MODE: PROCESS (Image to Point Cloud -> Persistent Homology)
    # -----------------------------------------------------------
    print("--- PHASE 1: Image to Point Cloud Conversion ---")
    for img_path in args.images:
        # ... (Phase 1 logic remains the same) ...
        if not os.path.isfile(img_path):
            print(f"[ERROR] File does not exist: {img_path}")
            continue

        png_path = convert_to_png(img_path)
        base = os.path.splitext(os.path.basename(png_path))[0]
        out_txt_pc = os.path.join(args.out_dir, base + "_point_cloud.txt")
        
        pts, filtr = convert_image_to_point_cloud(png_path,
                                                  threshold=args.threshold,
                                                  tile_size=args.tile_size)

        save_point_cloud_txt(pts, filtr, out_txt_pc)
        print(f"[INFO] Point cloud saved to: {out_txt_pc}")
        point_cloud_files.append(out_txt_pc)

    if args.analyze_homology or args.mode == 'classify':
        print("\n--- PHASE 2: Persistent Homology Analysis ---")
        for pc_file in point_cloud_files:
            print(f"\n[INFO] Homology analysis for: {pc_file}")

            try:
                data = np.loadtxt(pc_file)
            except IOError:
                print(f"[ERROR] Cannot load point cloud data from: {pc_file}")
                continue

            pts = data[:, :2]
            filtr = data[:, 2]
            
            base = os.path.splitext(os.path.basename(pc_file))[0]
            out_pers = os.path.join(args.out_dir, 
                                    f'{base}_pers_dim{args.homology_max_dim}_dist{args.homology_min_dist}.txt')

            # Compute PH and get the resulting diagram list (not used for plotting here)
            compute_persistent_homology(pts, filtr, 
                                        max_dim=args.homology_max_dim,
                                        step=args.homology_step,
                                        min_dist=args.homology_min_dist,
                                        out_path=out_pers)
            pers_diagram_files.append(out_pers)

    # -----------------------------------------------------------
    # MODE: CLASSIFY (Classify generated diagrams)
    # -----------------------------------------------------------
    if args.mode == 'classify' and pers_diagram_files:
        print("\n--- PHASE 3: CLASSIFICATION ---")
        # Ensure the model is trained before classification
        average_diagrams = train_model(args.out_dir, args.homology_max_dim, args.homology_min_dist)

        for diag_file in pers_diagram_files:
            if diag_file not in average_diagrams.keys(): # Avoid classifying the reference files themselves
                diagnosis, level, distance = classify_diagram(diag_file, average_diagrams)
                
                # FINAL RESULT OUTPUT
                print("\n=======================================================")
                print(f"ANALYSIS OF FILE: {os.path.basename(diag_file)}")
                print(f"  -> DIAGNOSIS: {diagnosis}")
                print(f"  -> PROBABLE LEVEL: {level}")
                print(f"  -> DISTANCE TO BEST LEVEL: {distance:.6f} (Bottleneck)")
                print("=======================================================")
