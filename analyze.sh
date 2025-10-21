#!/bin/bash

# --- 1. Konfiguracja (ZMIEN TO) ---
# Włóż swoje pliki .tiff do katalogu 'data/'
# Pliki do analizy:
# Zmień "data/unknown_prostate.tiff" na ścieżki do Twoich plików, np. "data/0.tiff data/1.tiff"
INPUT_FILES="data/unknown_prostate.tiff"
# Plik referencyjny B (musisz go wcześniej wygenerować z obrazu NORMALNEGO)
# Przykład: results/normal_prostate_point_cloud_pers_dim1_dist10.0.txt
# Upewnij się, że ta ścieżka jest poprawna.
REFERENCE_FILE="results/normal_prostate_pc_pers_dim1_dist10.0.txt"


# --- 2. Ustawienia TDA ---
# max_dim=1 dla pętli (H1), min_dist=10.0 do odfiltrowania szumu.
MAX_DIM=1 
MIN_DIST=10.0 
# threshold: Piksele ciemniejsze niż 180 (0-255) są brane pod uwagę (jądra komórkowe).
THRESHOLD=180
# step: Próbkowanie punktów (np. co 10. punkt) dla szybszych obliczeń PH.
HOMOLOGY_STEP=10

# --- 3. Instalacja zależności ---
echo "Instalowanie zależności..."
pip install -r requirements.txt
echo "Instalacja zakończona."

# --- 4. Uruchomienie analizy TDA ---
echo "Uruchamianie analizy TDA..."

# Krok 1 i 2: Konwersja, Homologia Persystentna i generacja PD
python3 TDA_Analysis.py ${INPUT_FILES} \
    --threshold ${THRESHOLD} \
    --analyze-homology \
    --homology-step ${HOMOLOGY_STEP} \
    --homology-max-dim ${MAX_DIM} \
    --homology-min-dist ${MIN_DIST} \
    --out-dir results

# Zbieranie nazw wygenerowanych plików diagramów dla porównania
# Zakładamy, że generowane pliki mają nazwy w formacie:
# results/plik_point_cloud_pers_dim{MAX_DIM}_dist{MIN_DIST}.txt
GENERATED_FILES=""
for file in ${INPUT_FILES}; do
    # Usuwamy ścieżkę i rozszerzenie, by uzyskać samą nazwę pliku
    BASE_NAME=$(basename "${file%.*}")
    GENERATED_FILES+="results/${BASE_NAME}_point_cloud_pers_dim${MAX_DIM}_dist${MIN_DIST}.txt "
done

# Krok 3: Porównanie diagramów
echo ""
echo "--- Porównanie Diagramów Persystencji ---"
python3 TDA_Analysis.py ${GENERATED_FILES} \
    --compare-diagrams \
    --diagram-b-file ${REFERENCE_FILE} \
    --diagram-dim ${MAX_DIM}

echo ""
echo "Analiza zakończona. Sprawdź katalog 'results/'."
