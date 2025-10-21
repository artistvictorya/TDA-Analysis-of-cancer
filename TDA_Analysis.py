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

def convert_to_png(image_path):
    """Converts the image (e.g., TIFF) to a grayscale PNG format."""
    if image_path.lower().endswith(".png"):
        return image_path

    png_path = os.path.splitext(image_path)[0] + ".png"
    print(f"[INFO] Converting {image_path} to {png_path}")

    with Image.open(image_path) as img:
        # Convert to grayscale (L - luminance)
        img.convert("L").save(png_path)

    return png_path

def convert_image_to_point_cloud(image_path, threshold=None, tile_size=1000):
    """Creates a point cloud (x, y, z) from the image, where z is the filtration value (brightness)."""
    with Image.open(image_path) as img:
        w, h = img.size
        print(f"[DEBUG] image size: {w} x {h}, mode: {img.mode}")

        points = []

        # Process with tiling for very large images
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tile_w = min(tile_size, w - x)
                tile_h = min(tile_size, h - y)
                tile = img.crop((x, y, x + tile_w, y + tile_h))
                arr = np.array(tile, dtype=np.uint8)

                # Generate coordinates and filtration values
                xs, ys = np.meshgrid(np.arange(x, x + tile_w), np.arange(y, y + tile_h), indexing='xy')
                tile_points = np.column_stack((xs.ravel(), ys.ravel(), arr.ravel()))

                # Brightness filtering (only pixels < threshold)
                if threshold is not None:
                    # Low brightness values (dark) correspond to cell nuclei or structures
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
    """Computes Persistent Homology using Delaunay Triangulation."""
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

        # 3) Propagate filtration: max over faces
        pbar.set_description("3) Propagating Filtration")
        # Use propagation so that the filtration of edges and faces is the maximum of the filtration of their vertices/faces
        max_dim_tree = st.dimension()
        for dim in range(1, max_dim_tree + 1):
            for simplex, _ in st.get_skeleton(dim):
                if len(simplex) != dim + 1:
                    continue
                face_filtrs = [st.filtration(list(face))
                               for face in itertools.combinations(simplex, dim)]
                st.assign_filtration(simplex, max(face_filtrs))
        st.initialize_filtration()
        pbar.update(1)

        # 4) Compute persistence
        pbar.set_description("4) Computing Persistence")
        raw_pers = st.persistence()
        pbar.update(1)

        # 5) Filter & Save result
        pbar.set_description("5) Saving and Filtering")
        pers = [(dim, (b, d)) for dim, (b, d) in raw_pers if dim <= max_dim and d < float('inf')]
        
        # Filter features close to the diagonal (short-lived - noise)
        # Note: abs(d-b) / sqrt(2) is the distance from the diagonal in L-infinity. We use min_dist as the persistence length threshold.
        filtered = [(dim, (b, d)) for dim, (b, d) in pers
                    if abs(d - b) >= min_dist] # Filter by persistence length (death - birth)

        with open(out_path, 'w') as f:
            f.write("# dim birth death\n")
            for dim, (b, d) in filtered:
                f.write(f"{dim} {b:.6f} {d:.6f}\n")
        pbar.update(1)

        # 6) Plot diagram
        pbar.set_description("6) Plotting Diagram")
        plt.figure(figsize=(6, 6))
        gudhi.plot_persistence_diagram(filtered)
        plt.title(f'Persistence Diagram (dim ≤ {max_dim}, persistence ≥ {min_dist})')
        plt.xlabel('Birth (Brightness)')
        plt.ylabel('Death (Brightness)')
        
        # Save plot as PNG
        plot_out_path = os.path.splitext(out_path)[0] + '.png'
        plt.savefig(plot_out_path)
        plt.close()
        pbar.update(1)

    print(f"[INFO] Persistence Diagram saved to: {out_path}")
    print(f"[INFO] Plot saved to: {plot_out_path}")


def read_diagram(filename, dim=None):
    """Loads a persistence diagram from a text file."""
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

def compare_diagrams(a_files, b_file, dim):
    """Compares diagrams from set A to diagram B using Bottleneck Distance."""
    print(f"\n[INFO] Comparing diagrams in dimension {dim}...")
    
    if not os.path.isfile(b_file):
        print(f"Error: B-file not found: {b_file}", file=sys.stderr)
        return

    diag_b = read_diagram(b_file, dim)
    if not diag_b:
        print(f"Warning: No points found in dimension {dim} in B-file", file=sys.stderr)
        return

    best_file = None
    best_dist = None

    for a in a_files:
        if not os.path.isfile(a):
            print(f"Warning: A-file not found: {a}", file=sys.stderr)
            continue
        
        # Skip B file if it is in set A
        try:
            if os.path.samefile(a, b_file):
                print(f"[INFO] Skipping B-file in set A: {a}")
                continue
        except Exception:
            pass

        diag_a = read_diagram(a, dim)
        if not diag_a:
            print(f"Warning: No points found in dimension {dim} in A-file: {a}", file=sys.stderr)
            continue

        try:
            dist = gudhi.bottleneck_distance(diag_a, diag_b)
        except Exception as e:
            print(f"Error computing distance for {a}: {e}", file=sys.stderr)
            continue

        print(f"Distance({os.path.basename(a)}, {os.path.basename(b_file)}) = {dist:.6f}")
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_file = a

    if best_file is None:
        print("No comparable diagram found in set A.")
    else:
        print(f"\nClosest diagram to {os.path.basename(b_file)} is: {os.path.basename(best_file)} (distance = {best_dist:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description="Topological Data Analysis (TDA) for Histopathological Images"
    )
    parser.add_argument('images', nargs='+',
                        help='Paths to image files (TIFF, PNG, JPG) to process.')
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
                        
    # Arguments for Diagram Comparison
    parser.add_argument('--compare-diagrams', action='store_true',
                        help='Compare persistence diagrams using Bottleneck Distance.')
    parser.add_argument('--diagram-b-file',
                        help='Reference (B) persistence diagram file for comparison.')
    parser.add_argument('--diagram-dim', type=int, default=1,
                        help='Homological dimension for diagram comparison (e.g., 1 for loops). Default is 1.')


    args = parser.parse_args()

    # Create folders
    os.makedirs(args.out_dir, exist_ok=True)
    point_cloud_files = []
    pers_diagram_files = []

    # PHASE 1: Image to Point Cloud Conversion
    print("--- PHASE 1: Image to Point Cloud Conversion ---")
    for img_path in args.images:
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


    # PHASE 2: Persistent Homology Analysis
    if args.analyze_homology:
        print("\n--- PHASE 2: Persistent Homology Analysis ---")
        for pc_file in point_cloud_files:
            print(f"\n[INFO] Homology analysis for: {pc_file}")

            data = np.loadtxt(pc_file)
            pts = data[:, :2]
            filtr = data[:, 2]
            
            base = os.path.splitext(os.path.basename(pc_file))[0]
            out_pers = os.path.join(args.out_dir, 
                                    f'{base}_pers_dim{args.homology_max_dim}_dist{args.homology_min_dist}.txt')

            compute_persistent_homology(pts, filtr, 
                                        max_dim=args.homology_max_dim,
                                        step=args.homology_step,
                                        min_dist=args.homology_min_dist,
                                        out_path=out_pers)
            pers_diagram_files.append(out_pers)


    # PHASE 3: Diagram Comparison
    if args.compare_diagrams:
        if not args.diagram_b_file:
            # Use the first generated diagram as the reference file (B)
            if pers_diagram_files:
                args.diagram_b_file = pers_diagram_files[0]
                print(f"[INFO] Using {args.diagram_b_file} as the reference (B) file.")
            else:
                print("Error: No diagrams to compare. Run the script with --analyze-homology and provide a B-file.", file=sys.stderr)
                sys.exit(1)

        print("\n--- PHASE 3: Comparing Persistence Diagrams ---")
        compare_diagrams(pers_diagram_files, args.diagram_b_file, args.diagram_dim)
