# Design Spec - Reorganizing Project Directory Structure

We will reorganize the codebase directory structure to group notebooks, source code, models, and data logically.

## Proposed Structure

```text
Simple-OCR/
├── notebooks/                  # Jupyter Notebooks (.ipynb)
│   ├── Simple-OCR.ipynb
│   ├── Simple-OCR_runpod.ipynb
│   └── Simple_OCR_colab.ipynb
├── src/                        # Python Source Code (renamed from Python/)
│   ├── main.py
│   ├── model.py
│   ├── model2.py
│   ├── model_colab.py
│   └── detection.py
├── models/                     # Model weights and checkpoints (.h5, .keras)
│   ├── OCR Model.h5
│   ├── OCR_Model_Best.keras
│   └── ...
├── data/                       # Datasets
│   ├── curated/
│   └── ETL/
├── requirements.txt
└── .gitignore
```

## Detailed Relocations

1. **Notebooks**:
   - Move `Simple-OCR.ipynb`, `Simple-OCR_runpod.ipynb`, `Simple_OCR_colab.ipynb` to `notebooks/`.
2. **Python Sources**:
   - Move `model.py`, `model2.py`, `model_colab.py` (which are currently duplicated in root) and all contents of `Python/` into `src/`.
   - Delete the old `Python/` directory.
3. **Models**:
   - Move `OCR Model.h5`, `OCR Model_base.h5`, `OCR_Model_Best.keras`, and `Python/OCR-2U-024.h5` into `models/`.
4. **Data**:
   - Move `ETL/` and `curated/` into a new `data/` directory.
   
## Required Code Modifications

1. **Jupyter Notebooks**:
   - Since notebooks are moved to `notebooks/`, add `sys.path.append('../src')` at the beginning of the imports cell so they can load modules from `src/`.
   - Update `dataset_path` and `etl1_dir` pathing in notebooks to locate datasets relative to the new `notebooks/` directory (e.g., `../data/curated` and `../data/ETL/ETL1`).
2. **Main Python Script (`src/main.py`)**:
   - Update relative imports or data pathing if they assume execution from root.
3. **Gitignore (`.gitignore`)**:
   - Update `.gitignore` paths to match new directory layout (e.g. `data/`, `models/`).
