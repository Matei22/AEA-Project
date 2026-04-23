import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Set
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import coo_matrix
from tqdm import tqdm
from sgn import SparseGraphNetwork


class NeuroLKHTrainer:
    def __init__(self, model: SparseGraphNetwork, lr: float = 1e-4, eta_pi: float = 1.0):
        self.model = model
        self.eta_pi = eta_pi
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    
    def edge_loss(self, beta: torch.Tensor, edge_index: torch.Tensor, 
                  optimal_edges: Set) -> torch.Tensor:
        src, tgt = edge_index[0], edge_index[1]
        
        is_optimal = torch.tensor([
            1.0 if (src[i].item(), tgt[i].item()) in optimal_edges else 0.0
            for i in range(len(src))
        ], device=beta.device, dtype=torch.float32)
        
        beta_clamped = torch.clamp(beta, min=1e-7, max=1.0 - 1e-7)
        
        loss = -(
            is_optimal * torch.log(beta_clamped) +
            (1 - is_optimal) * torch.log(1 - beta_clamped)
        )
        
        return loss.mean()
    
    def node_loss(self, pi: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
        return -torch.mean((degrees - 2) * pi)
    
    def compute_1tree_degrees(self, coords: torch.Tensor, pi: torch.Tensor, 
                               special_node: int = 0) -> torch.Tensor:
        N = coords.shape[0]
        
        pi_clamped = torch.clamp(pi, min=-10.0, max=10.0)
        
        dist = torch.cdist(coords, coords)
        transformed = dist + pi_clamped.unsqueeze(0) + pi_clamped.unsqueeze(1)
        transformed_np = transformed.cpu().detach().numpy()
        transformed_np = np.maximum(transformed_np, 0)
        
        other_nodes = [i for i in range(N) if i != special_node]
        if len(other_nodes) <= 1:
            return torch.ones(N, device=coords.device) * 2
        
        subgraph = transformed_np[other_nodes][:, other_nodes]
        
        try:
            mst = minimum_spanning_tree(subgraph)
            mst_coo = coo_matrix(mst)
            
            degrees = np.zeros(N)
            for i, j, d in zip(mst_coo.row, mst_coo.col, mst_coo.data):
                if d > 0:
                    u, v = other_nodes[i], other_nodes[j]
                    degrees[u] += 1
                    degrees[v] += 1
            
            special_dist = transformed_np[special_node].copy()
            special_dist[special_node] = np.inf
            nearest = np.argsort(special_dist)[:2]
            for n in nearest:
                if special_dist[n] < np.inf:
                    degrees[special_node] += 1
                    degrees[n] += 1
        except:
            degrees = np.ones(N) * 2
        
        return torch.tensor(degrees, device=coords.device, dtype=torch.float32)
    
    def train_step(self, coords: torch.Tensor, optimal_tour: List[int]) -> dict:
        self.model.train()
        
        optimal_edges = set()
        for i in range(len(optimal_tour)):
            u, v = optimal_tour[i], optimal_tour[(i + 1) % len(optimal_tour)]
            optimal_edges.add((u, v))
            optimal_edges.add((v, u))
        
        beta, pi, edge_index, _ = self.model(coords)
        
        if torch.isnan(beta).any() or torch.isnan(pi).any():
            return {'loss': float('nan'), 'beta_loss': float('nan'), 'pi_loss': float('nan')}
        
        beta_loss = self.edge_loss(beta, edge_index, optimal_edges)
        degrees = self.compute_1tree_degrees(coords, pi)
        pi_loss = self.node_loss(pi, degrees)
        
        total_loss = beta_loss + self.eta_pi * pi_loss
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return {'loss': float('nan'), 'beta_loss': float('nan'), 'pi_loss': float('nan')}
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        grad_norm = 0.0
        has_nan_grad = False
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break
                grad_norm += p.grad.norm().item() ** 2
        
        if has_nan_grad:
            self.optimizer.zero_grad()
            return {'loss': float('nan'), 'beta_loss': float('nan'), 'pi_loss': float('nan')}
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'beta_loss': beta_loss.item(),
            'pi_loss': pi_loss.item()
        }


def train(
    data_path: str,
    save_dir: str,
    epochs: int = 16,
    device: str = 'cuda',
    lr: float = 1e-4
):
    print(f"Loading training data from {data_path}...")
    with open(data_path, 'rb') as f:
        training_data = pickle.load(f)
    print(f"Loaded {len(training_data)} training instances")
    
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SparseGraphNetwork(D=128, L=30, gamma=20, C=10.0)  
    model.to(device)
    
    trainer = NeuroLKHTrainer(model, lr=lr, eta_pi=1.0)  
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {epochs} epochs (lr={lr})...")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_beta_loss = 0.0
        epoch_pi_loss = 0.0
        valid_steps = 0
        skipped_steps = 0
        
        np.random.shuffle(training_data)
        
        pbar = tqdm(training_data, desc=f"Epoch {epoch+1}/{epochs}")
        for instance in pbar:
            coords = torch.tensor(instance['coords'], dtype=torch.float32, device=device)
            tour = instance['tour']
            
            losses = trainer.train_step(coords, tour)
            
            if not np.isnan(losses['loss']):
                epoch_loss += losses['loss']
                epoch_beta_loss += losses['beta_loss']
                epoch_pi_loss += losses['pi_loss']
                valid_steps += 1
                pbar.set_postfix({'loss': f"{losses['loss']:.4f}"})
            else:
                skipped_steps += 1
                pbar.set_postfix({'loss': 'skip'})
        
        if valid_steps > 0:
            avg_loss = epoch_loss / valid_steps
            avg_beta = epoch_beta_loss / valid_steps
            avg_pi = epoch_pi_loss / valid_steps
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Beta={avg_beta:.4f}, Pi={avg_pi:.4f} ({valid_steps}/{len(training_data)} steps, {skipped_steps} skipped)")
             
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_path / 'neurolkh_best.pt')
        else:
            print(f"Epoch {epoch+1}: All steps skipped! Reducing learning rate.")
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"New LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': avg_loss if valid_steps > 0 else float('inf'),
        }, save_path / f'checkpoint_epoch_{epoch+1}.pt')
    
    final_model = save_path / 'neurolkh_final.pt'
    if not final_model.exists():
        torch.save(model.state_dict(), final_model)
    print(f"\nModel saved to {save_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NeuroLKH')
    parser.add_argument('--data', type=str, default='training_data/training_data.pkl')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args.data, args.save_dir, args.epochs, args.device, args.lr)