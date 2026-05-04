"""Per-molecule factor assignment via a chain conditional random field.

Mirrors R ``run_crf_all`` / ``run_crf``. Inference uses loopy belief propagation
over the molecule-KNN graph; node potentials come from NMF gene loadings,
edge potentials from a single tunable label-agreement parameter.

Implementation in JAX: the union of all per-cell KNN graphs is treated as one
disconnected graph (cells share no edges), so a single JIT-compiled BP routine
runs over all molecules in parallel — no per-cell loop, no padding. Max-product
(MAP) decoding.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, Float, Int

from .nmf import KNNNMFResult, project_nmf
from .utils import (
    cells_with_min_molecules,
    gene_index,
    knn_indices_xyz,
    normalize_loadings,
    validate_transcripts,
)


def build_molecule_graph(
    df: pd.DataFrame, h: int
) -> tuple[Int[np.ndarray, " e2"], Int[np.ndarray, " e2"], Int[np.ndarray, " e2"], np.ndarray]:
    """Build the union of per-cell molecule-KNN graphs as one disconnected graph.

    For each cell with ``> h`` molecules, take the symmetric ``h``-NN graph in
    3D coordinates; concatenate all cells' edge lists with a global molecule
    offset. Each undirected edge ``{i, j}`` is stored as two directed entries
    in consecutive positions, so ``reverse_idx[e] = e ^ 1``.

    Parameters
    ----------
    df
        Transcript-level dataframe.
    h
        KNN neighbourhood size.

    Returns
    -------
    senders, receivers
        Two ``int32`` arrays of length ``2 * n_undirected_edges`` with directed
        edges ``senders[e] -> receivers[e]``.
    reverse_idx
        For each directed edge, the index of its reverse: ``reverse_idx[e] = e ^ 1``.
    mol_ids
        Molecule ids in the order used as global node indices (cells with
        ``<= h`` molecules are excluded).
    """
    keep_cells = cells_with_min_molecules(df, h + 1)
    df_kept = df[df["cell"].isin(keep_cells)]
    if df_kept.empty:
        raise ValueError(f"No cells with > {h} molecules.")

    senders_all: list[np.ndarray] = []
    receivers_all: list[np.ndarray] = []
    mol_ids_all: list[np.ndarray] = []
    offset = 0
    for _, df_cell in df_kept.groupby("cell", sort=False):
        coords = df_cell[["x", "y", "z"]].to_numpy(dtype=np.float64)
        n = coords.shape[0]
        nn_idx = knn_indices_xyz(coords, k=h, include_self=False)
        local_senders = np.repeat(np.arange(n), nn_idx.shape[1])
        local_receivers = nn_idx.reshape(-1)
        a = np.minimum(local_senders, local_receivers)
        b = np.maximum(local_senders, local_receivers)
        unique = np.unique(np.stack([a, b], axis=1), axis=0)
        senders_all.append(unique[:, 0] + offset)
        receivers_all.append(unique[:, 1] + offset)
        mol_ids_all.append(df_cell["mol_id"].to_numpy())
        offset += n

    u_senders = np.concatenate(senders_all)
    u_receivers = np.concatenate(receivers_all)
    n_undirected = u_senders.shape[0]
    senders = np.empty(2 * n_undirected, dtype=np.int32)
    receivers = np.empty(2 * n_undirected, dtype=np.int32)
    senders[0::2] = u_senders
    receivers[0::2] = u_receivers
    senders[1::2] = u_receivers
    receivers[1::2] = u_senders
    reverse_idx = np.arange(2 * n_undirected, dtype=np.int32) ^ 1
    return senders, receivers, reverse_idx, np.concatenate(mol_ids_all)


def molecule_node_potentials(
    df: pd.DataFrame,
    mol_ids: np.ndarray,
    H_norm: Float[Array, "k g"],
    gene_names: np.ndarray,
) -> Float[Array, "n k"]:
    """Build per-molecule node potentials by gene-indexing into ``H_norm``.

    Returns ``(n_mols, k)`` where ``out[i, f] = H_norm[f, gene_idx_of_molecule_i]``.

    Genes present in ``df`` but missing from ``gene_names`` are assigned the
    minimum value in ``H_norm`` (matches R fallback in ``run_crf_all``).
    """
    df_idx = df.set_index("mol_id").loc[mol_ids]
    gene_idx, _ = gene_index(df_idx.reset_index(), all_genes=gene_names)
    H_full = jnp.asarray(H_norm)
    H_full_T = H_full.T  # (g, k)
    return H_full_T[gene_idx]


@partial(jax.jit, static_argnames=("n_iter",))
def loopy_bp_max_product(
    log_node_pot: Float[Array, "n k"],
    log_edge_pot: Float[Array, "k k"],
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
    reverse_idx: Int[Array, " e"],
    n_iter: int,
) -> Int[Array, " n"]:
    """Max-product loopy belief propagation; return MAP labels.

    All inputs are ``jnp`` arrays. The graph is given as a directed edge list
    where each undirected edge contributes two directed entries; ``reverse_idx``
    maps a directed edge to its opposite. The standard max-product update is

        m_{i→j}(x_j) = max_{x_i} [ log_node_pot[i, x_i]
                                  + sum_{k in N(i)} m_{k→i}(x_i)
                                  - m_{j→i}(x_i)
                                  + log_edge_pot[x_i, x_j] ]

    Messages are normalised to zero max each iteration for stability. Edge
    potentials are assumed symmetric and shared across edges (matches the
    cellAdmix model where all edges use the same transition matrix ``A``).
    """
    n_nodes, k = log_node_pot.shape
    n_edges = senders.shape[0]
    log_msg = jnp.zeros((n_edges, k))

    def step(log_msg, _):
        # Sum of all incoming messages at each node, then add node potentials.
        log_belief = log_node_pot.at[receivers].add(log_msg)
        # For each directed edge i→j, subtract the back-message j→i so we sum
        # only over neighbours of i other than j.
        incoming_minus_back = log_belief[senders] - log_msg[reverse_idx]
        # New message: max_{x_i} [ incoming_minus_back[e, x_i] + log_edge_pot[x_i, x_j] ]
        new_msg = jnp.max(
            incoming_minus_back[:, :, None] + log_edge_pot[None, :, :],
            axis=1,
        )
        new_msg = new_msg - jnp.max(new_msg, axis=1, keepdims=True)
        return new_msg, None

    log_msg, _ = jax.lax.scan(step, log_msg, None, length=n_iter)
    log_belief = log_node_pot.at[receivers].add(log_msg)
    return jnp.argmax(log_belief, axis=1)


def run_crf_all(
    df: pd.DataFrame,
    res: KNNNMFResult,
    num_nn: int = 10,
    same_label_ratio: float = 5.0,
    normalize_by: str = "gene",
    n_iter: int = 200,
    proj_h: int | None = None,
) -> pd.DataFrame:
    """Assign each molecule to one of the ``k`` NMF factors via loopy BP.

    Mirrors R ``run_crf_all``.

    Parameters
    ----------
    df
        Transcript-level dataframe.
    res
        Output of :func:`pycelladmix.nmf.run_knn_nmf`. If
        ``res.gene_subset`` is ``True``, gene loadings are projected onto the
        full gene set via :func:`pycelladmix.nmf.project_nmf` before BP.
    num_nn
        KNN neighbourhood size for the molecule graph (R default 10).
    same_label_ratio
        Ratio of agreement to disagreement edge potential.
        ``t = 1 / (same_label_ratio + k - 1)``; the edge potential matrix
        ``A`` then has ``1 - (k-1) t`` on the diagonal and ``t`` off-diagonal
        (R default 5).
    normalize_by
        ``"gene"`` (default), ``"factor"``, or ``"gene.factor"`` — see
        :func:`pycelladmix.utils.normalize_loadings`.
    n_iter
        Loopy-BP iterations (R default 200).
    proj_h
        Neighbourhood size used for ``project_nmf`` if loading projection is
        needed. Defaults to ``2 * num_nn`` (matches R).

    Returns
    -------
    DataFrame with columns ``mol_id`` and ``factor`` (1-indexed factor
    assignments to mirror R) in the row order of ``df`` after dropping
    molecules in cells with too few neighbours.
    """
    validate_transcripts(df)

    if res.gene_subset:
        proj_h = proj_h if proj_h is not None else 2 * num_nn
        H_full, gene_names_full = project_nmf(df, res, h=proj_h)
    else:
        H_full = res.H
        gene_names_full = res.gene_names

    H_norm = normalize_loadings(H_full, by=normalize_by)
    H_np = np.asarray(H_norm)
    if (H_np <= 0).any():
        floor = float(H_np[H_np > 0].min())
        H_norm = jnp.where(H_norm > 0, H_norm, floor)

    senders, receivers, reverse_idx, mol_ids = build_molecule_graph(df, h=num_nn)
    node_pot = molecule_node_potentials(df, mol_ids, H_norm, gene_names_full)
    log_node_pot = jnp.log(node_pot + 1e-30)

    k = node_pot.shape[1]
    t = 1.0 / (same_label_ratio + k - 1)
    A = jnp.full((k, k), t).at[jnp.diag_indices(k)].set(1.0 - (k - 1) * t)
    log_edge_pot = jnp.log(A + 1e-30)

    labels = loopy_bp_max_product(
        log_node_pot,
        log_edge_pot,
        jnp.asarray(senders),
        jnp.asarray(receivers),
        jnp.asarray(reverse_idx),
        n_iter=n_iter,
    )
    labels_np = np.asarray(labels) + 1  # 1-indexed to match R

    return pd.DataFrame({"mol_id": mol_ids, "factor": labels_np})


__all__ = [
    "build_molecule_graph",
    "loopy_bp_max_product",
    "molecule_node_potentials",
    "run_crf_all",
]
