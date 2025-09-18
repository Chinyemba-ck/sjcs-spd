"""Notebook clustering implementation ported from Final pipeline/SPD_Pipeline_Colab.ipynb

This module contains the exact clustering code from the research notebooks,
adapted for use in the comparison interface.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Literal
import warnings

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import SpectralClustering

from spd.models.component_model import ComponentModel


@dataclass
class SPDComponent:
    """Represents a single SPD component with its associated data."""
    layer: str
    component_index: int
    U: np.ndarray  # Output direction vector (U matrix row from LinearComponent)
    V: np.ndarray  # Input direction vector (V matrix column from LinearComponent)
    g_profile: Optional[np.ndarray] = None
    causal_importance: Optional[float] = None


@dataclass
class NotebookClusteringResults:
    """Results from notebook clustering implementation."""
    labels: np.ndarray
    n_clusters: int
    cluster_sizes: Dict[int, int]
    similarity_matrix: np.ndarray
    components: List[SPDComponent]
    similarity_method: str
    total_components: int
    
    def get_cluster_members(self, cluster_id: int) -> List[int]:
        """Get component indices for a specific cluster."""
        return [i for i, label in enumerate(self.labels) if label == cluster_id]

    def to_json(self) -> dict:
        """Export notebook clustering results to JSON-compatible dictionary.

        Returns complete clustering data including:
        - Cluster assignments for each component
        - Cluster sizes
        - Component information (layer, index)
        - Similarity method used
        - Total component count
        """
        # Build detailed cluster membership information
        cluster_details = {}
        for cluster_id in range(self.n_clusters):
            members = self.get_cluster_members(cluster_id)
            cluster_details[str(cluster_id)] = {
                "member_indices": members,
                "member_labels": [self.components[i].layer + f":{self.components[i].component_index}"
                                  for i in members],
                "size": len(members)
            }

        # Build component details
        component_details = []
        for i, comp in enumerate(self.components):
            component_details.append({
                "index": i,
                "layer": comp.layer,
                "component_index": comp.component_index,
                "cluster_id": int(self.labels[i]),
                # Optionally include causal importance if available
                "causal_importance": float(comp.causal_importance) if comp.causal_importance is not None else None
            })

        # Convert numpy arrays to lists for JSON serialization
        labels_list = self.labels.tolist() if hasattr(self.labels, 'tolist') else list(self.labels)

        # Note: Not including similarity_matrix as it can be very large
        # If needed, it can be added with: "similarity_matrix": self.similarity_matrix.tolist()

        # Convert cluster_sizes keys and values from numpy types to Python int for JSON serialization
        cluster_sizes_json = {int(k): int(v) for k, v in self.cluster_sizes.items()}

        return {
            "n_clusters": self.n_clusters,
            "total_components": self.total_components,
            "similarity_method": self.similarity_method,
            "cluster_sizes": cluster_sizes_json,
            "cluster_labels": labels_list,
            "cluster_details": cluster_details,
            "component_details": component_details
        }


def extract_spd_components(
    comp_model: ComponentModel,
    components: Dict[str, Any],
    gates: Dict[str, Any],
    n_samples: int = 1000,
    min_active_features: int = 1,
    max_active_features: int = 3,
    compute_gate_profiles: bool = True,
    device: str = 'cpu',
    random_seed: Optional[int] = None
) -> Tuple[List[SPDComponent], np.ndarray]:
    """Extract SPD components from a trained ComponentModel.
    
    Ported exactly from Final pipeline notebook cell 4.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    comp_model = comp_model.to(device)
    comp_model.eval()

    n_features = comp_model.patched_model.config.n_features

    # Generate sample batch
    batch = torch.zeros(n_samples, n_features, device=device)
    for i in range(n_samples):
        n_active = torch.randint(min_active_features, max_active_features + 1, (1,)).item()
        active_indices = torch.randperm(n_features)[:n_active]
        batch[i, active_indices] = torch.rand(n_active, device=device)

    spd_components = []

    for layer_name, component in components.items():
        A_matrix = component.V.detach().cpu().numpy()  # (d_in, C) - V matrix is the input transformation
        B_matrix = component.U.detach().cpu().numpy()  # (C, d_out) - U matrix is the output transformation

        gate_outputs = None
        if compute_gate_profiles:
            gate = gates[layer_name]

            with torch.no_grad():
                _, pre_weight_acts = comp_model._forward_with_pre_forward_cache_hooks(
                    batch, module_names=[layer_name]
                )

            layer_input = pre_weight_acts[layer_name]
            component_acts = torch.einsum('bd,dc->bc', layer_input, component.V)
            gate_outputs = gate(component_acts.detach()).detach().cpu().numpy()

        n_components = A_matrix.shape[1]
        for c in range(n_components):
            spd_comp = SPDComponent(
                layer=layer_name,
                component_index=c,
                U=B_matrix[c, :],          # d_out dimensional vector (output direction)
                V=A_matrix[:, c],          # d_in dimensional vector (input direction)
                g_profile=gate_outputs[:, c] if gate_outputs is not None else None
            )
            spd_components.append(spd_comp)

    return spd_components, batch.cpu().numpy()


def directional_similarity(
    c1: SPDComponent,
    c2: SPDComponent,
    epsilon: float = 1e-9
) -> float:
    """Compute directional cosine similarity between component input vectors.
    
    Ported exactly from Final pipeline notebook cell 6.
    """
    v1, v2 = c1.V, c2.V

    if v1.shape != v2.shape:
        raise ValueError(f"Dimension mismatch: {v1.shape} vs {v2.shape}")

    numerator = float(np.dot(v1, v2))
    denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2) + epsilon)

    return numerator / denominator


def activation_correlation(
    c1: SPDComponent,
    c2: SPDComponent,
    epsilon: float = 1e-9
) -> float:
    """Compute Pearson correlation between component activation profiles.
    
    Ported exactly from Final pipeline notebook cell 6.
    """
    if c1.g_profile is None or c2.g_profile is None:
        raise ValueError("Both components must have gate profiles for correlation")

    x, y = c1.g_profile, c2.g_profile

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    x_std = x.std() + epsilon
    y_std = y.std() + epsilon

    x_normalized = x_centered / x_std
    y_normalized = y_centered / y_std

    n = len(x)
    correlation = float(np.dot(x_normalized, y_normalized) / (n - 1 + epsilon))

    return np.clip(correlation, -1.0, 1.0)


def coactivation_expectation(
    c1: SPDComponent,
    c2: SPDComponent
) -> float:
    """Compute co-activation expectation: E_x[g_i(x) * g_j(x)]."""
    if c1.g_profile is None or c2.g_profile is None:
        raise ValueError("Both components must have gate profiles for coactivation")
    
    return float(np.mean(c1.g_profile * c2.g_profile))


def compute_fused_similarity(
    components: List[SPDComponent],
    alpha: float = 0.6,
    beta: float = 0.25,
    gamma: float = 0.15,
    normalize: bool = True
) -> np.ndarray:
    """Compute weighted combination of multiple similarity metrics.
    
    Ported exactly from Final pipeline notebook cell 6.
    """
    if normalize:
        total = alpha + beta + gamma
        if abs(total) < 1e-9:
            raise ValueError("Weights sum to zero")
        alpha, beta, gamma = alpha/total, beta/total, gamma/total

    n = len(components)
    S_fused = np.zeros((n, n))

    has_profiles = all(c.g_profile is not None for c in components)
    if not has_profiles and (beta > 0 or gamma > 0):
        warnings.warn("Components lack gate profiles; using directional similarity only")
        alpha, beta, gamma = 1.0, 0.0, 0.0

    for i in range(n):
        for j in range(i, n):
            if i == j:
                S_fused[i, j] = 1.0
            else:
                sim = 0
                # Only compute directional similarity if components have matching V dimensions
                # This is mathematically required - cosine similarity is undefined for vectors of different dimensions
                if alpha > 0 and components[i].V.shape == components[j].V.shape:
                    sim += alpha * directional_similarity(components[i], components[j])
                if beta > 0 and has_profiles:
                    sim += beta * activation_correlation(components[i], components[j])
                if gamma > 0 and has_profiles:
                    sim += gamma * coactivation_expectation(components[i], components[j])
                S_fused[i, j] = S_fused[j, i] = sim

    return S_fused


def cross_layer_clustering(
    components: List[SPDComponent],
    n_clusters: Optional[int] = None,
    similarity_weights: Optional[Dict[str, float]] = None,
    k_neighbors: int = 8,
    random_state: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """Main function for cross-layer clustering of SPD components.
    
    Ported exactly from Final pipeline notebook cell 6.
    """
    if similarity_weights is None:
        similarity_weights = {
            'directional': 0.6,
            'correlation': 0.25,
            'coactivation': 0.15
        }

    if verbose:
        print(f"Starting cross-layer clustering with {len(components)} components...")

    similarity_matrix = compute_fused_similarity(
        components,
        alpha=similarity_weights.get('directional', 0.6),
        beta=similarity_weights.get('correlation', 0.25),
        gamma=similarity_weights.get('coactivation', 0.15)
    )

    # Auto-detect clusters if not specified
    if n_clusters is None:
        n_clusters = min(len(components) // 5, 10)  # Simple heuristic
        n_clusters = max(2, n_clusters)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state
    )

    labels = clustering.fit_predict(np.clip(similarity_matrix, 0, 1))

    # Analysis
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes_dict = dict(zip(unique_labels, counts))

    analysis = {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes_dict,
        'similarity_matrix': similarity_matrix
    }

    if verbose:
        print(f"Clustering complete: {n_clusters} clusters found")
        print(f"Cluster sizes: {analysis['cluster_sizes']}")

    return labels, analysis


def run_notebook_clustering(
    component_model: ComponentModel,
    similarity_method: Literal["fused", "directional", "correlation", "coactivation"] = "fused",
    n_clusters: Optional[int] = None,
    n_samples: int = 1000,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> NotebookClusteringResults:
    """Run the complete notebook clustering pipeline.
    
    Args:
        component_model: Trained ComponentModel
        similarity_method: Which similarity metric to use
        n_clusters: Number of clusters (auto-detect if None)
        n_samples: Number of samples for gate profile computation
        random_state: Random seed for reproducibility
        verbose: Print progress messages
        
    Returns:
        NotebookClusteringResults with all clustering information
    """
    # Extract components and gates from ComponentModel
    components = {
        k.removeprefix("components.").replace("-", "."): v
        for k, v in component_model.components.items()
    }
    gates = {
        k.removeprefix("gates.").replace("-", "."): v
        for k, v in component_model.gates.items()
    }
    
    # Extract SPD components with gate profiles
    spd_components, _ = extract_spd_components(
        component_model, 
        components, 
        gates,
        n_samples=n_samples,
        compute_gate_profiles=True,
        random_seed=random_state
    )
    
    if len(spd_components) == 0:
        raise ValueError("No components extracted from model")
    
    # Set up similarity weights based on method
    if similarity_method == "fused":
        similarity_weights = {'directional': 0.6, 'correlation': 0.25, 'coactivation': 0.15}
    elif similarity_method == "directional":
        similarity_weights = {'directional': 1.0, 'correlation': 0.0, 'coactivation': 0.0}
    elif similarity_method == "correlation":
        similarity_weights = {'directional': 0.0, 'correlation': 1.0, 'coactivation': 0.0}
    elif similarity_method == "coactivation":
        similarity_weights = {'directional': 0.0, 'correlation': 0.0, 'coactivation': 1.0}
    else:
        raise ValueError(f"Unknown similarity method: {similarity_method}")
    
    # Run clustering
    labels, analysis = cross_layer_clustering(
        spd_components,
        n_clusters=n_clusters,
        similarity_weights=similarity_weights,
        random_state=random_state,
        verbose=verbose
    )
    
    return NotebookClusteringResults(
        labels=labels,
        n_clusters=analysis['n_clusters'],
        cluster_sizes=analysis['cluster_sizes'],
        similarity_matrix=analysis['similarity_matrix'],
        components=spd_components,
        similarity_method=similarity_method,
        total_components=len(spd_components)
    )