import torch
import numpy as np
import os
import time
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sgn import SparseGraphNetwork


@dataclass
class Solution:
    tour: List[int]
    distance: float
    time: float
    n_trials: int


class NeuroLKHSolver:
    def __init__(
            self,
            model_path: str,
            lkh_executable: str = "LKH-3.0.14/LKH-3.exe",
            gamma: int = 20,
            k: int = 5,
            n_trials: int = None,
            device: str = "cuda"
    ):
        self.gamma = gamma
        self.k = k
        self.n_trials = n_trials
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.lkh_executable = Path(lkh_executable)

        if not self.lkh_executable.exists():
            raise FileNotFoundError(
                f"LKH executable not found at {lkh_executable}\n"
                "Download LKH from: http://www.akira.ruc.dk/~keld/research/LKH-3/"
            )

        print(f"Using LKH executable: {self.lkh_executable}")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = SparseGraphNetwork(gamma=gamma)
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded model from {model_path}")

    def solve(self, coords, problem=None, run_id=0, verbose=False) -> Solution:
        start_time = time.time()
        n_nodes = len(coords)
        trials = self.n_trials if self.n_trials else n_nodes

        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            beta, pi, edge_index, _ = self.model(coords_tensor)

        transformed_dist = self.model.transform_distances(coords_tensor, pi)
        transformed_dist_np = transformed_dist.cpu().numpy()

        candidates = self.model.get_candidates(beta, edge_index, k=self.k)

        if verbose:
            print(f"  SGN inference: {time.time() - start_time:.3f}s")

        tour, distance = self._run_lkh_executable(
            coords=coords,
            transformed_dist=transformed_dist_np,
            candidates=candidates,
            n_nodes=n_nodes,
            n_trials=trials,
            run_id=run_id,
            problem=problem
        )

        elapsed = time.time() - start_time
        return Solution(tour=tour, distance=distance, time=elapsed, n_trials=trials)

    def _run_lkh_executable(
            self,
            coords: np.ndarray,
            transformed_dist: np.ndarray,
            candidates: List[Tuple[int, int, int]],
            n_nodes: int,
            n_trials: int,
            run_id: int,
            problem=None
    ) -> Tuple[List[int], float]:
        dist_matrix = np.array(transformed_dist, dtype=np.float64)
        dist_matrix = np.maximum(dist_matrix, 0.0)
        np.fill_diagonal(dist_matrix, 0.0)

        scale = 1e6 / (dist_matrix.max() + 1e-10)
        int_dist = np.round(dist_matrix * scale).astype(np.int64)
        np.fill_diagonal(int_dist, 0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsp", delete=False, encoding="utf-8") as f:
            tsp_file = f.name
            f.write(f"NAME: neurolkh_temp_{run_id}\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {n_nodes}\n")
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for i in range(n_nodes):
                f.write(" ".join(str(int_dist[i, j]) for j in range(n_nodes)))
                f.write("\n")
            f.write("EOF\n")

        cand_file = None
        if candidates:
            cand_file = tsp_file.replace(".tsp", ".cand")
            node_candidates = {i: [] for i in range(n_nodes)}
            for from_node, to_node, priority in candidates:
                node_candidates[from_node].append((to_node, int(priority)))

            with open(cand_file, "w", encoding="utf-8") as cf:
                cf.write(f"{n_nodes}\n")
                for i in range(n_nodes):
                    node_list = node_candidates[i]
                    node_list.sort(key=lambda x: x[1])
                    edge_str = " ".join(f"{to + 1} {pri}" for to, pri in node_list)
                    cf.write(f"{i + 1} 0 {len(node_list)} {edge_str}\n")

        par_file = tsp_file.replace(".tsp", ".par")
        with open(par_file, "w", encoding="utf-8") as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\n")
            f.write(f"MAX_TRIALS = {n_trials}\n")
            f.write("RUNS = 1\n")
            f.write(f"SEED = {run_id}\n")
            f.write("TRACE_LEVEL = 0\n")
            f.write(f"OUTPUT_TOUR_FILE = {tsp_file}.tour\n")
            if cand_file:
                f.write("MAX_CANDIDATES = 0\n")
                f.write(f"CANDIDATE_FILE = {cand_file}\n")

        result = subprocess.run(
            [str(self.lkh_executable), par_file],
            capture_output=True,
            timeout=3600,
            text=True
        )

        if result.returncode != 0:
            self._cleanup_files(tsp_file, par_file, cand_file)
            raise RuntimeError(f"LKH failed with code {result.returncode}:\n{result.stderr}")

        tour_file = tsp_file + ".tour"
        tour = self._read_lkh_tour(tour_file, n_nodes)

        if tour is None:
            self._cleanup_files(tsp_file, par_file, cand_file)
            raise RuntimeError("Failed to read LKH tour file")

        distance = self._calculate_tour_distance(tour, coords, problem)

        self._cleanup_files(tsp_file, par_file, cand_file, tour_file)
        return tour, distance

    def _cleanup_files(self, *files):
        for f in files:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass

    def _read_lkh_tour(self, tour_file: str, n_nodes: int) -> Optional[List[int]]:
        if not os.path.exists(tour_file):
            return None

        tour = []
        with open(tour_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        in_tour = False
        for line in lines:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_tour = True
                continue
            if line in ["-1", "EOF"]:
                break
            if in_tour:
                try:
                    node = int(line)
                    if 1 <= node <= n_nodes:
                        tour.append(node - 1)
                except ValueError:
                    continue

        if len(tour) != n_nodes:
            return None

        return tour

    def _calculate_tour_distance(self, tour, coords, problem=None):
        if problem is not None:
            nodes = sorted(problem.get_nodes())
            idx_to_node = {i: nodes[i] for i in range(len(nodes))}
            total = 0.0
            for i in range(len(tour)):
                u = idx_to_node[tour[i]]
                v = idx_to_node[tour[(i + 1) % len(tour)]]
                total += problem.get_weight(u, v)
            return float(total)

        distance = 0.0
        for i in range(len(tour)):
            u, v = tour[i], tour[(i + 1) % len(tour)]
            distance += np.linalg.norm(coords[u] - coords[v])
        return float(distance)