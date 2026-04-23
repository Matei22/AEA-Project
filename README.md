# NeuroLKH-inspired TSP Solver

This project implements a NeuroLKH-style hybrid solver for the Traveling Salesman Problem (TSP).

It combines:
- a **Sparse Graph Network (SGN)** that predicts:
  - **edge scores** `beta`, used to build a candidate set
  - **node penalties** `pi`, used to transform edge distances
- the **LKH solver**, which performs the final local search

The implementation is inspired by the NeuroLKH paper, but it is **not a perfect reproduction** of the original method.

## Project structure

```text
neurolkh/
├── benchmark_instances/
│   ├── a280.tsp
│   ├── ali535.tsp
│   ├── att48.tsp
│   ├── ...
│   ├── ulysses22.tsp
│   ├── usa13509.tsp
│   ├── vm1084.tsp
│   └── vm1748.tsp
├── checkpoints/
│   └── neurolkh_best.pt
├── LKH-3.0.14/
│   └── LKH-3.exe
├── training_data/
│   └── training_data.pkl
├── benchmark.py
├── data_generator.py
├── run.py
├── sgn.py
├── solution.txt
├── solver.py
└── train.py
```

## Requirements

- Python 3.10+ recommended
- PyTorch
- NumPy
- SciPy
- tqdm
- tsplib95
- LKH executable

## 1. Create and activate a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

## 2. Install dependencies

### CPU version

```bash
pip install torch numpy scipy tqdm tsplib95 pandas
```

### CUDA version

Example for CUDA-enabled PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy scipy tqdm tsplib95 pandas matplotlib seaborn
```

You can verify the installation with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 3. Prepare LKH

Place the LKH executable at:

```text
LKH-3.0.14/LKH-3.exe
```

If your executable is elsewhere, pass it explicitly with `--lkh`.

## 4. Training data

If you already have:

```text
training_data/training_data.pkl
```

you do **not** need to regenerate the dataset.

This file is used directly by `train.py`.

Each training sample contains:
- `coords`
- `tour`
- `distance`
- `size`
- `type`

## 5. Train the model

From the `neurolkh/` folder:

### GPU
```bash
python train.py --data training_data/training_data.pkl --save_dir checkpoints --epochs 16 --device cuda --lr 1e-4
```

Main output files:
- `checkpoints/neurolkh_best.pt`
- `checkpoints/checkpoint_epoch_X.pt`
- `checkpoints/neurolkh_final.pt`

## 6. Run on a single TSPLIB instance

Example:

```bash
python run.py --model checkpoints/neurolkh_best.pt --data benchmark_instances/berlin52.tsp --device cuda
```

## 7. Run the benchmark on a folder

```bash
python run.py --model checkpoints/neurolkh_best.pt --data benchmark_instances --device cuda
```

This runs the hybrid solver on all `.tsp` files that can be processed and saves a JSON summary in `results/`.


## 8. Notes about TSPLIB support

The benchmark pipeline was extended to work better with TSPLIB:
- it uses `tsplib95`
- it can evaluate instances with edge-weight types such as:
  - `EUC_2D`
  - `ATT`
  - `GEO`

However, the neural model still requires coordinates as input, so instances without usable coordinates may still be skipped.

## 9. Known differences from the original NeuroLKH paper

This project is inspired by the paper, but differs in some important ways:
- training labels were generated with **LKH**, not **Concorde**
- the implementation is simplified compared to the original paper
- some TSPLIB cases still require practical handling

## 10. Common issues

### Some benchmark instances are skipped
This usually means the instance does not provide usable coordinates for the neural model.

## 11. Quick start

If everything is already prepared, the shortest useful sequence is:

```bash
pip install torch numpy scipy tqdm tsplib95 pandas matplotlib seaborn
python train.py --data training_data/training_data.pkl --save_dir checkpoints --epochs 16 --device cuda --lr 1e-4
python run.py --model checkpoints/neurolkh_best.pt --data benchmark_instances/berlin52.tsp --device cuda
python run.py --model checkpoints/neurolkh_best.pt --data benchmark_instances --device cuda
```
