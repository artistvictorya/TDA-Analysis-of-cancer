#!/bin/bash

# --- 1. Configuration (CHANGE THIS) ---
# Place your .tiff files in the 'data/' directory
# Files to analyze:
INPUT_FILES="data/unknown_prostate.tiff"
# Reference B file (You must generate this beforehand from a NORMAL image)
# Example: results/normal_prostate_point_cloud_pers_dim1_dist10.0.txt
REFERENCE_FILE="results/normal_prostate_point_cloud_pers_dim1_dist10.0.txt"


# --- 2. TDA Settings ---
# max_dim=1 for loops (H1), min_dist=10.0 to filter noise.
MAX_DIM=1 
MIN_DIST=10.0 
# threshold: Pixels darker than 180 (0-255) are considered (cell nuclei).
THRESHOLD=180
# step: Point cloud sampling factor (e.g., every 10th point) for faster PH calculations.
HOMOLOGY_STEP=10

# --- 3. Install Dependencies ---
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Installation complete."

# --- 4. Run TDA Analysis ---
echo "Running TDA Analysis..."

# Step 1 and 2: Conversion, Persistent Homology, and PD generation
python3 TDA_Analysis.py ${INPUT_FILES} \
    --threshold ${THRESHOLD} \
    --analyze-homology \
    --homology-step ${HOMOLOGY_STEP} \
    --homology-max-dim ${MAX_DIM} \
    --homology-min-dist ${MIN_DIST} \
    --out-dir results

# Collect the names of the generated diagram files for comparison
GENERATED_FILES=""
for file in ${INPUT_FILES}; do
    BASE_NAME=$(basename "${file%.*}")
    GENERATED_FILES+="results/${BASE_NAME}_point_cloud_pers_dim${MAX_DIM}_dist${MIN_DIST}.txt "
done

# Step 3: Compare Diagrams
echo ""
echo "--- Comparing Persistence Diagrams ---"
python3 TDA_Analysis.py ${GENERATED_FILES} \
    --compare-diagrams \
    --diagram-b-file ${REFERENCE_FILE} \
    --diagram-dim ${MAX_DIM}

echo ""
echo "Analysis complete. Check the 'results/' directory."
