import numpy as np
import pickle
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


class TSPDataGenerator:    
    def __init__(self, lkh_executable: str = "LKH-3.0.14/LKH-3.exe"):
        self.lkh_executable = lkh_executable
        
        if not Path(lkh_executable).exists():
            raise FileNotFoundError(f"LKH executable not found at {lkh_executable}")
    
    def generate_uniform_instance(self, n_nodes: int) -> np.ndarray:

        return np.random.uniform(0, 1, (n_nodes, 2)).astype(np.float32)
    
    def generate_clustered_instance(self, n_nodes: int, n_clusters: int = None) -> np.ndarray:
        if n_clusters is None:
            n_clusters = np.random.randint(3, 9)
        
        coords = []
        nodes_per_cluster = n_nodes // n_clusters
        remaining = n_nodes % n_clusters
        
        for c in range(n_clusters):
            cluster_center = np.random.uniform(0.2, 0.8, 2)
            cluster_std = np.random.uniform(0.03, 0.08)
            
            n_this = nodes_per_cluster + (1 if c < remaining else 0)
            cluster_coords = np.random.normal(cluster_center, cluster_std, (n_this, 2))
            cluster_coords = np.clip(cluster_coords, 0, 1)
            coords.append(cluster_coords)
        
        coords = np.vstack(coords)
        np.random.shuffle(coords)

        return coords[:n_nodes].astype(np.float32)
    
    def solve_with_lkh(self, coords: np.ndarray, trials: int = 100) -> Tuple[List[int], float]:
        N = len(coords)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsp', delete=False) as f:
            tsp_file = f.name
            f.write(f"NAME: temp\nTYPE: TSP\nDIMENSION: {N}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n")

            for i, (x, y) in enumerate(coords):
                f.write(f"{i+1} {x} {y}\n")

            f.write("EOF\n")
        
        par_file = tsp_file.replace('.tsp', '.par')
        with open(par_file, 'w') as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\n")
            f.write(f"MAX_TRIALS = {trials}\n")
            f.write(f"TRACE_LEVEL = 0\n")
            f.write(f"OUTPUT_TOUR_FILE = {tsp_file}.tour\n")
        
        subprocess.run([self.lkh_executable, par_file], capture_output=True, timeout=300)
        
        tour_file = tsp_file + '.tour'
        tour = []
        with open(tour_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    node = int(line)
                    if 1 <= node <= N:
                        tour.append(node - 1)
        
        os.unlink(tsp_file)
        os.unlink(par_file)
        if os.path.exists(tour_file):
            os.unlink(tour_file)
        
        distance = sum(np.linalg.norm(coords[tour[i]] - coords[tour[(i+1)%N]]) for i in range(N))

        return tour, distance
    
    def generate_dataset(
        self,
        sizes: List[int],
        n_instances_per_size: int,
        output_dir: str,
        mix_distributions: bool = True
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_data = []
        
        for size in sizes:
            n_instances = max(int(n_instances_per_size / size * 100), 10)
            print(f"\nGenerating {n_instances} instances of size {size}...")
            
            for i in tqdm(range(n_instances), desc=f"Size {size}"):
                if mix_distributions and i % 2 == 0:
                    coords = self.generate_clustered_instance(size)
                    dist_type = 'clustered'
                else:
                    coords = self.generate_uniform_instance(size)
                    dist_type = 'uniform'
                
                tour, distance = self.solve_with_lkh(coords, trials=min(100, size))
                
                all_data.append({
                    'coords': coords,
                    'tour': tour,
                    'distance': distance,
                    'size': size,
                    'type': dist_type
                })
        
        output_file = output_path / 'training_data.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(all_data, f)
        
        print(f"\nSaved {len(all_data)} instances to {output_file}")

        return all_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate NeuroLKH training data')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 200, 300, 400, 500])
    parser.add_argument('--n_instances', type=int, default=500000)
    parser.add_argument('--output_dir', type=str, default='training_data')
    parser.add_argument('--lkh', type=str, default='LKH-3.0.14/LKH-3.exe')
    parser.add_argument('--no_mix', action='store_true')
    
    args = parser.parse_args()
    
    generator = TSPDataGenerator(lkh_executable=args.lkh)
    generator.generate_dataset(
        sizes=args.sizes,
        n_instances_per_size=args.n_instances,
        output_dir=args.output_dir,
        mix_distributions=not args.no_mix
    )