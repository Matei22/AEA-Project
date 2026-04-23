import sys
import argparse
from pathlib import Path
from solver import NeuroLKHSolver
from benchmark import run_benchmark, load_tsplib_instance


def main():
    parser = argparse.ArgumentParser(description='NeuroLKH TSP Solver')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--lkh', type=str, default='LKH-3.0.14/LKH-3.exe', help='Path to LKH executable')
    parser.add_argument('--data', type=str, required=True, help='TSPLIB directory or single .tsp file')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per instance')
    parser.add_argument('--trials', type=int, default=None, help='LKH trials (default: N)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    solver = NeuroLKHSolver(
        model_path=args.model,
        lkh_executable=args.lkh,
        n_trials=args.trials,
        device=args.device
    )
    
    data_path = Path(args.data)
    
    if data_path.is_file() and data_path.suffix == '.tsp':
        print(f"\nLoading {data_path.name}...")
        coords, opt_dist = load_tsplib_instance(str(data_path))
        print(f"Loaded {len(coords)} nodes")
        if opt_dist:
            print(f"Optimal distance: {opt_dist:.2f}")
        
        print("\nSolving...")
        sol = solver.solve(coords, verbose=True)
        
        print(f"\n{'='*50}")
        print(f"Result: {sol.distance:.2f}")
        if opt_dist:
            gap = 100 * (sol.distance - opt_dist) / opt_dist
            print(f"Optimal: {opt_dist:.2f}")
            print(f"Gap: {gap:.4f}%")
        print(f"Time: {sol.time:.2f}s")
        print(f"Trials: {sol.n_trials}")
        print(f"{'='*50}")
        
    elif data_path.is_dir():
        print(f"\nRunning benchmark on {data_path}...")
        results = run_benchmark(
            solver, 
            str(data_path), 
            runs_per_instance=args.runs,
            output_dir=args.output
        )
        print(f"\nTested {len(results)} instances")
    else:
        print(f"Error: {args.data} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()