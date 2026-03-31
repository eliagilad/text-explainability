from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, FrozenSet, List, Sequence

import numpy as np
import torch
from shap.maskers import Text


def _import_hedge() -> object:
    """
    Import HEDGE's `hedge_bert.py` as a module by adding its directory to `sys.path`.
    We do this to avoid calling the CLI entrypoint (`hedge_main_bert_imdb_debug.py`).
    """
    import sys

    hedge_dir = None
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "external_repos" / "HEDGE" / "bert"
        if candidate.exists():
            hedge_dir = candidate
            break
    if hedge_dir is None:
        raise ModuleNotFoundError(
            "Could not locate HEDGE at external_repos/HEDGE/bert "
            f"from {Path(__file__).resolve()}"
        )

    if str(hedge_dir) not in sys.path:
        sys.path.insert(0, str(hedge_dir))

    import hedge_bert as hedge  # type: ignore

    return hedge


def _ensure_token_type_ids(enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
    return enc


def _tokens_from_shap_segments(segments: Sequence[str]) -> List[str]:
    """
    SHAP's Text masker stores segments that are close to (but not exactly) wordpieces.
    We mirror your existing balanced/random logic: add '##' to continuation pieces.
    """
    import re

    tokens: List[str] = []
    space_end = re.compile(r"^.*\W$")
    letter_start = re.compile(r"^[A-Za-z]")
    for i, v in enumerate(segments):
        v = v.strip()
        if i > 0 and space_end.match(segments[i - 1]) is None and letter_start.match(v) is not None and tokens[i - 1] != "":
            tokens.append("##" + v)
        else:
            tokens.append(v)
    return tokens


def _hedge_partitions_for_tokens(
    *,
    tokens: Sequence[str],
    model,
    tokenizer,
    win_size: int,
    max_length: int,
) -> List[List[List[int]]]:
    hedge = _import_hedge()

    device = next(model.parameters()).device
    model.eval()

    # Force HEDGE to see exactly the same leaf tokens SHAP will mask.
    # This avoids linkage/segment length mismatches.
    tokens = list(tokens)[:max_length]
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer must define cls_token_id and sep_token_id for HEDGE.")

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([[cls_id] + token_ids + [sep_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    token_type_ids = torch.zeros_like(input_ids, device=device)

    # HEDGE expects `labels` to exist because it indexes logits as `model(**inputs)[1]`
    # (HF models return `(loss, logits, ...)` when labels are provided).
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": torch.zeros((1,), dtype=torch.long, device=device),
    }

    args = SimpleNamespace(device=device)
    expl = hedge.HEDGE(model, inputs, args, thre=100)
    expl.compute_shapley_hier_tree(model, inputs, win_size)

    # `expl.hier_tree[level]` is a list of `(subset, score)` pairs; we only need subsets.
    partitions: List[List[List[int]]] = []
    for level in sorted(expl.hier_tree.keys()):
        partitions.append([list(subset) for (subset, _score) in expl.hier_tree[level]])

    return partitions


def _partitions_to_linkage(partitions: Sequence[Sequence[Sequence[int]]]) -> np.ndarray:
    """
    Convert HEDGE's top-down partitions to a SciPy-like linkage matrix:
      (M-1) x 4 with columns [idx1, idx2, height, cluster_size].

    HEDGE partitions increase the number of clusters by 1 each level (one split into two).
    We convert by walking levels backwards and turning each split into a merge.
    """
    if len(partitions) < 2:
        raise ValueError("Need at least 2 partition levels to build linkage.")

    last_level = [frozenset(x) for x in partitions[-1]]
    all_items = sorted(set().union(*last_level))
    if not all_items:
        raise ValueError("Empty partition tree.")
    M = len(all_items)

    # Map clusters (as sets of leaf indices) -> linkage index.
    cluster_id: Dict[FrozenSet[int], int] = {frozenset([i]): i for i in range(M)}

    linkage = np.zeros((M - 1, 4), dtype=float)
    next_id = M
    row = 0

    for lvl in range(len(partitions) - 1, 0, -1):
        prev = {frozenset(x) for x in partitions[lvl - 1]}
        curr = {frozenset(x) for x in partitions[lvl]}

        added = list(curr - prev)     # two children created by the split
        removed = list(prev - curr)   # the parent cluster before split

        if len(added) != 2 or len(removed) != 1:
            raise ValueError(
                f"Unexpected partition delta at level {lvl}: "
                f"added={len(added)}, removed={len(removed)}"
            )

        a, b = added
        parent = removed[0]
        if a | b != parent:
            raise ValueError(f"Level {lvl} is not a binary split of a single parent cluster.")

        ida = cluster_id[a]
        idb = cluster_id[b]
        size = float(len(parent))
        height = size  # will be normalized later (SHAP only needs relative heights)

        linkage[row, 0] = ida
        linkage[row, 1] = idb
        linkage[row, 2] = height
        linkage[row, 3] = size

        # After merging in reverse, parent becomes a new cluster id.
        cluster_id[parent] = next_id
        next_id += 1
        row += 1

    if row != M - 1:
        raise ValueError(f"Built {row} merges, expected {M - 1}.")

    # Normalize heights to [0, 1] for stability (matches your other maskers).
    if linkage[:, 2].max() > 0:
        linkage[:, 2] = linkage[:, 2] / linkage[:, 2].max()

    return linkage


class InteractionTextMasker(Text):
    """
    A SHAP Text masker whose token clustering is produced by HEDGE's interaction-based tree.
    """

    def __init__(
        self,
        tokenizer,
        model,
        *,
        win_size: int = 1,
        max_length: int = 250,
        mask_token=None,
        collapse_mask_token: str = "auto",
        output_type: str = "string",
    ):
        super().__init__(tokenizer, mask_token, collapse_mask_token, output_type)
        self.model = model
        self.win_size = win_size
        self.max_length = max_length

    def clustering(self, s: str) -> np.ndarray:
        self._update_s_cache(s)
        tokens = _tokens_from_shap_segments(self._segments_s)
        partitions = _hedge_partitions_for_tokens(
            tokens=tokens,
            model=self.model,
            tokenizer=self.tokenizer,
            win_size=self.win_size,
            max_length=self.max_length,
        )
        linkage = _partitions_to_linkage(partitions)
        linkage = np.asarray(linkage, dtype=float)
        if linkage.ndim != 2 or linkage.shape[1] != 4:
            raise ValueError(f"Invalid linkage shape {linkage.shape}, expected (M-1, 4).")
        return linkage