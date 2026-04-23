import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
import tsplib95


@dataclass
class BenchmarkResult:
    instance: str
    n_nodes: int
    runs: int
    best_distance: float
    mean_distance: float
    std_distance: float
    optimal_distance: Optional[float]
    optimal_gap: Optional[float]
    mean_time: float
    success_rate: float


def load_tsplib_instance(filepath: str) -> Tuple[object, Optional[np.ndarray], Optional[float]]:
    """
    Load a TSPLIB instance with tsplib95.

    Returns:
        problem: tsplib95 problem object
        coords: Nx2 float32 array if coordinates exist, else None
        optimal_distance: BEST_KNOWN / OPTIMAL_VALUE if present, else None
    """
    problem = tsplib95.load(filepath)

    coords = None
    if getattr(problem, "node_coords", None):
        nodes = sorted(problem.node_coords.keys())
        coords = np.array([problem.node_coords[n] for n in nodes], dtype=np.float32)

    optimal_distance = None
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if "BEST_KNOWN" in line or "OPTIMAL_VALUE" in line:
                    parts = line.replace(":", " ").split()
                    for token in reversed(parts):
                        try:
                            optimal_distance = float(token)
                            break
                        except ValueError:
                            pass
                    if optimal_distance is not None:
                        break
    except Exception:
        pass

    return problem, coords, optimal_distance


def run_benchmark(
        solver,
        data_dir: str,
        runs_per_instance: int = 10,
        output_dir: str = "results",
        verbose: bool = True
) -> Dict[str, BenchmarkResult]:
    data_path = Path(data_dir)
    instance_files = sorted(data_path.glob("*.tsp"))

    if not instance_files:
        print(f"No .tsp files found in {data_dir}")
        return {}

    results = {}

    for tsp_file in instance_files:
        instance_name = tsp_file.stem
        problem, coords, optimal_dist = load_tsplib_instance(str(tsp_file))
        n_nodes = problem.dimension

        if coords is None:
            print(f"Skipping {instance_name}: no coordinates available for the model")
            continue

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Testing {instance_name} ({n_nodes} nodes)")
            print(f"EDGE_WEIGHT_TYPE: {getattr(problem, 'edge_weight_type', 'UNKNOWN')}")
            if optimal_dist is not None:
                print(f"Best known: {optimal_dist:.2f}")
            print("=" * 60)

        distances = []
        times = []
        optimal_found = 0

        for run in range(runs_per_instance):
            sol = solver.solve(coords, problem=problem, run_id=run, verbose=False)

            distances.append(sol.distance)
            times.append(sol.time)

            if optimal_dist is not None and abs(sol.distance - optimal_dist) < 1e-6:
                optimal_found += 1

            if verbose:
                gap_str = ""
                if optimal_dist is not None:
                    gap = 100.0 * (sol.distance - optimal_dist) / optimal_dist
                    gap_str = f" (gap: {gap:.4f}%)"
                print(f"  Run {run + 1:2d}: {sol.distance:.2f}{gap_str} [{sol.time:.2f}s]")

        distances = np.array(distances, dtype=np.float64)

        result = BenchmarkResult(
            instance=instance_name,
            n_nodes=n_nodes,
            runs=runs_per_instance,
            best_distance=float(np.min(distances)),
            mean_distance=float(np.mean(distances)),
            std_distance=float(np.std(distances)),
            optimal_distance=optimal_dist,
            optimal_gap=(
                100.0 * (float(np.min(distances)) - optimal_dist) / optimal_dist
                if optimal_dist is not None else None
            ),
            mean_time=float(np.mean(times)),
            success_rate=optimal_found / runs_per_instance if optimal_dist is not None else 0.0
        )

        results[instance_name] = result

    print_summary(results)
    save_results(results, output_dir)

    return results


def print_summary(results: Dict[str, BenchmarkResult]):
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Instance':<20} {'Nodes':>6} {'Best':>12} {'Optimal':>12} {'Gap%':>10} {'Time(s)':>10}")
    print("-" * 90)

    for name, r in sorted(results.items()):
        gap_str = f"{r.optimal_gap:.4f}" if r.optimal_gap is not None else "N/A"
        opt_str = f"{r.optimal_distance:.2f}" if r.optimal_distance is not None else "N/A"
        print(
            f"{name:<20} {r.n_nodes:>6} {r.best_distance:>12.2f} "
            f"{opt_str:>12} {gap_str:>10} {r.mean_time:>10.2f}"
        )

    print("=" * 90)


def save_results(results: Dict[str, BenchmarkResult], output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"benchmark_{timestamp}.json"

    output_data = {name: asdict(r) for name, r in results.items()}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_file}")