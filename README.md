# TDA-Analysis-of-cancer
# 🔬 Prostate Cancer Analysis Using Persistent Homology

This project uses Topological Data Analysis (TDA), specifically Persistent Homology, to extract topological features (shape) from histopathological images (TIFF) of the prostate.

## How to Run in GitHub Codespaces

1.  **Clone** this repository into a Codespace.
2.  **Add Image Files**: Place your `.tiff` files (e.g., prostate biopsies) into the `data/` directory.
3.  **Create/Add a Reference File**: To perform classification, you need a **Reference File** (persistence diagram) for a defined state (e.g., `normal_prostate_point_cloud_pers_dim1_dist10.0.txt`).
    * **Important**: This file must be generated beforehand from an image with a **known diagnosis** (e.g., healthy tissue or a specific Gleason Score).
4.  **Configuration**: Open the `analyze.sh` file and update the `--- 1. Configuration ---` section with the paths to your files and the TDA settings.
    * Ensure the **`REFERENCE_FILE`** points to the pre-generated diagram (or modify `INPUT_FILES` so the first file is your desired reference, although providing a static reference is better).
5.  **Run the Script**: In the Codespace terminal, execute:
    ```bash
    chmod +x analyze.sh
    ./analyze.sh
    ```

## 📈 Interpreting the Results

The script will generate files in the `results/` directory:
* `*_point_cloud.txt`: The point cloud data (x, y, brightness).
* `*_pers_dim1_dist10.0.txt`: The Persistence Diagram (Birth/Death/Dimension).
* `*_pers_dim1_dist10.0.png`: The visualization of the diagram.

**Classification Criterion (Bottleneck Distance):**

The main output is the **Bottleneck Distance** calculated between the diagram of your new image and the **reference diagram (B)**.

* A **Low Bottleneck Distance** (e.g., below a defined threshold) means that the **topology (shape of glands)** of the analyzed image is **similar** to the topology of the reference image.
* If the reference diagram represents **healthy tissue**, a low distance might suggest the sample is also healthy.
* If the reference diagram represents **cancer (e.g., Gleason 4+3)**, a low distance might suggest similar aggressiveness.
* A **High Bottleneck Distance** indicates a **significant difference** in topology, which may suggest a different grade of cancer or a lack of cancer (depending on the reference).

**Persistence Diagram Interpretation:**

In the context of prostate cancer histopathology, we are interested in dimension $\mathbf{H_1}$ (loops/holes):
* **Loops ($H_1$)** in the Persistence Diagram correspond to **glandular structures** in the image.
* Long persistence (a large distance from the diagonal) of $H_1$ loops signifies **clear, stable glandular structures**.
* In prostate cancer (higher Gleason Score), these structures become **more irregular, fused, or disappear**. This leads to:
    * **Fewer** long-lived loops (fewer distinct, circular glands).
    * The appearance of loops with **lower persistence** (shorter-lived, irregular shapes).

**Conclusion:** By comparing the PD of an unknown sample against the PD of a **healthy reference tissue**, a larger Bottleneck Distance likely indicates **cancerous changes** (tumor tissue).

---

Would you like me to generate a set of **simulated diagram files** (one "Normal" reference and one "Cancerous" test file) so you can test the `analyze.sh` and `TDA_Analysis.py` comparison functionality right away?
