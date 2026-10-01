"""How LALS is defined, from the caption corpus to an image-level score.

1. Lexicon           female / male terms and the content-word filter
2. Reference corpus  46,227 gender-balanced COCO captions
3. Text database     contextual token embeddings of the corpus at one layer, with k-NN search
4. LALS              base rates, the score of a visual token, image-level aggregation

For a visual token at layer L, take its k = 20 nearest entries (cosine similarity) in the
layer-L text database and keep those whose own token is a content word. With f (m) the
fraction of kept entries whose caption contains a female (male) term,

    LALS = (f - f_base) - (m - m_base),

where f_base and m_base are the same fractions over all content-word entries of the
database. Positive = female-leaning, negative = male-leaning; a token without content-word
neighbours scores 0; the first visual token (attention sink) is excluded.

Hidden-state extraction from the VLMs (GPU) is in ``utils/models.py``.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------
# 1. Lexicon (paper Table 2)
# ---------------------------------------------------------------------------------------------

FEMALE_TERMS = frozenset({
    "woman", "girl", "female", "she", "her", "mother", "wife",
    "daughter", "sister", "lady", "feminine", "pregnant", "actress",
    "heroine", "women", "girls", "ladies", "mothers", "wives",
    "daughters", "sisters",
})
MALE_TERMS = frozenset({
    "man", "boy", "male", "he", "him", "his", "father", "husband",
    "son", "brother", "gentleman", "masculine", "actor", "hero",
    "men", "boys", "gentlemen", "fathers", "husbands", "sons", "brothers",
})
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "and", "or", "is", "are", "was", "were", "be", "been",
})


def caption_words(caption: str) -> set[str]:
    """Lower-cased alphabetic words of a caption."""
    return set(re.findall(r"[a-z]+", caption.lower()))


def is_content_word(token_str: str) -> bool:
    """More than 2 letters after stripping non-letters, and not a function word."""
    clean = re.sub(r"[^a-z]", "", token_str.strip().lower())
    return len(clean) > 2 and clean not in FUNCTION_WORDS


# ---------------------------------------------------------------------------------------------
# 2. Reference corpus
# ---------------------------------------------------------------------------------------------

def load_coco_captions(annotation_file: str | Path, max_sentences: int | None = 50_000,
                       seed: int = 42) -> list[str]:
    """Unique captions (``str.strip()``, first occurrence, file order), randomly subsampled.

    With COCO ``captions_train2017.json`` (569,002 unique captions) the captions at
    ``RandomState(seed).permutation(n)[:max_sentences]`` are kept, in that order.
    """
    annotations = json.loads(Path(annotation_file).read_text(encoding="utf-8"))["annotations"]
    captions = list(dict.fromkeys(a["caption"].strip() for a in annotations))
    if max_sentences is not None and len(captions) > max_sentences:
        order = np.random.RandomState(seed).permutation(len(captions))[:max_sentences]
        captions = [captions[i] for i in order]
    return captions


def gender_bucket(caption: str) -> str:
    """``female_only``, ``male_only``, ``both`` or ``neutral``."""
    words = caption_words(caption)
    female, male = not words.isdisjoint(FEMALE_TERMS), not words.isdisjoint(MALE_TERMS)
    return "both" if female and male else "female_only" if female else "male_only" if male else "neutral"


def gender_balance_sentences(captions: list[str], seed: int = 42) -> list[str]:
    """Downsample male-only captions to the number of female-only captions, then shuffle.

    One ``RandomState(seed)`` drives both steps: the kept male-only captions are
    ``sorted(rng.choice(n_male_only, n_female_only, replace=False))`` (input order is kept),
    then female-only + male-only + both + neutral is shuffled with the same generator.
    The paper's 50,000 captions give 46,227.
    """
    buckets: dict[str, list[str]] = {"female_only": [], "male_only": [], "both": [], "neutral": []}
    for caption in captions:
        buckets[gender_bucket(caption)].append(caption)
    rng = np.random.RandomState(seed)
    female_only, male_only = buckets["female_only"], buckets["male_only"]
    if len(male_only) > len(female_only):
        keep = np.sort(rng.choice(len(male_only), size=len(female_only), replace=False))
        male_only = [male_only[i] for i in keep]
    balanced = female_only + male_only + buckets["both"] + buckets["neutral"]
    rng.shuffle(balanced)
    return balanced


# ---------------------------------------------------------------------------------------------
# 3. Text database
# ---------------------------------------------------------------------------------------------
# Every caption is run through the VLM's language backbone (no image). At each layer, every
# token except special tokens and the first remaining token becomes one entry
# (hidden state, {token_str, token_id, caption, position}). On disk (LatentLens format):
#     <db_dir>/layer_<L>/embeddings_cache.pt = {"embeddings": float16 [N, dim],
#                                               "metadata": [...], "token_to_indices": {...}}

def layer_path(db_dir: str | Path, layer: int) -> Path:
    return Path(db_dir) / f"layer_{layer}" / "embeddings_cache.pt"


def _unit_rows(x):
    """Rows of a torch tensor scaled to unit length (float32), so that dot product = cosine."""
    x = x.float()
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


class TextDB:
    """One layer of the text database, with cosine k-NN search.

    ``exact`` compares each query with every entry (float16 on GPU; the layer sweeps).
    ``ivf`` is the approximate search of a FAISS ``IndexIVFFlat`` (the token maps and the colour
    experiments): FAISS k-means splits the entries into 256 clusters, and each query is only
    compared with the entries of its ``nprobe`` most similar clusters (float32).
    """

    N_CLUSTERS = 256

    def __init__(self, embeddings, metadata: list[dict]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError(f"{len(embeddings)} embeddings vs {len(metadata)} metadata rows")
        self.embeddings = embeddings  # torch tensor [N, dim]
        self.metadata = metadata
        self._unit = None      # unit-length embeddings, on the device and in the dtype of the last search
        self._clusters = None  # (centroids [256, dim], cluster of each entry [N]) for the ivf search

    @classmethod
    def load(cls, db_dir: str | Path, layer: int) -> "TextDB":
        import torch

        cache = torch.load(layer_path(db_dir, layer), map_location="cpu", weights_only=False)
        return cls(cache["embeddings"], cache["metadata"])

    def __len__(self) -> int:
        return len(self.metadata)

    def search(self, queries, k: int = 20, backend: str = "exact", nprobe: int = 32,
               device: str | None = None) -> np.ndarray:
        """Indices ``[n_queries, k]`` of the ``k`` most cosine-similar entries."""
        import torch

        if backend not in ("exact", "ivf"):
            raise ValueError(f"unknown backend {backend!r}")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if backend == "exact" and device.startswith("cuda") else torch.float32
        if self._unit is None or self._unit.device.type != torch.device(device).type or self._unit.dtype != dtype:
            self._unit = None  # free the previous copy first
            self._unit = _unit_rows(self.embeddings).to(device=device, dtype=dtype)
        q = _unit_rows(queries).to(device=device, dtype=dtype)
        sims = q @ self._unit.T
        if backend == "ivf":
            sims.masked_fill_(~self._probed(q, nprobe), float("-inf"))
        return sims.topk(k, dim=1).indices.cpu().numpy()

    def _probed(self, q, nprobe: int):
        """Boolean mask ``[n_queries, N]``: the entries of the ``nprobe`` clusters most similar to
        each query. Same neighbours as ``faiss.IndexIVFFlat.search`` (up to the order of tied
        entries), computed as a masked matrix product on the GPU instead of by FAISS on the CPU."""
        import torch

        if self._clusters is None:
            import faiss

            unit = np.ascontiguousarray(_unit_rows(self.embeddings).numpy())
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(unit.shape[1]), unit.shape[1], self.N_CLUSTERS,
                                       faiss.METRIC_INNER_PRODUCT)
            index.train(unit)  # FAISS k-means, fixed seed
            self._clusters = (torch.from_numpy(index.quantizer.reconstruct_n(0, self.N_CLUSTERS)),
                              torch.from_numpy(index.quantizer.assign(unit, 1).ravel()))
        centroids, cluster_of = (t.to(q.device) for t in self._clusters)
        nearest = (q @ centroids.T).topk(nprobe, dim=1).indices
        probed = torch.zeros(len(q), self.N_CLUSTERS, dtype=torch.bool, device=q.device)
        return probed.scatter_(1, nearest, True)[:, cluster_of]


def build_text_db(vlm, sentences: Sequence[str], layers: Sequence[int], out_dir: str | Path,
                  batch_size: int = 64) -> dict[int, Path]:
    """Embed ``sentences`` with ``vlm.text_token_states`` at ``layers`` and save one file per layer."""
    import torch
    from tqdm import tqdm

    embeddings: dict[int, list] = {layer: [] for layer in layers}
    metadata: list[dict] = []
    for start in tqdm(range(0, len(sentences), batch_size), desc="text DB", unit="batch"):
        states, meta = vlm.text_token_states(list(sentences[start:start + batch_size]), layers)
        for layer in layers:
            embeddings[layer].append(states[layer].to(torch.float16))
        metadata.extend(meta)

    token_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        token_to_indices[m["token_str"]].append(i)

    saved = {}
    for layer in layers:
        path = layer_path(out_dir, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": torch.cat(embeddings[layer]), "metadata": metadata,
                    "token_to_indices": dict(token_to_indices)}, path)
        saved[layer] = path
    return saved


# ---------------------------------------------------------------------------------------------
# 4. LALS
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseRates:
    female: float
    male: float


def token_lals(neighbours: list[dict], base: BaseRates, female_terms: frozenset[str] = FEMALE_TERMS,
               male_terms: frozenset[str] = MALE_TERMS) -> float:
    """LALS of one visual token from its k nearest database entries (reference definition).

    ``neighbours``: metadata dicts with ``token_str`` and ``caption``.
    """
    content = [n for n in neighbours if is_content_word(n["token_str"])]
    if not content:
        return 0.0
    f = sum(not caption_words(n["caption"]).isdisjoint(female_terms) for n in content) / len(content)
    m = sum(not caption_words(n["caption"]).isdisjoint(male_terms) for n in content) / len(content)
    return (f - base.female) - (m - base.male)


def label_entries(metadata: list[dict], female_terms: frozenset[str] = FEMALE_TERMS,
                  male_terms: frozenset[str] = MALE_TERMS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean arrays ``(is_content, is_female, is_male)`` over the database entries."""
    labels: dict[str, tuple[bool, bool]] = {}
    for m in metadata:
        if m["caption"] not in labels:
            words = caption_words(m["caption"])
            labels[m["caption"]] = (not words.isdisjoint(female_terms), not words.isdisjoint(male_terms))
    is_content = np.array([is_content_word(m["token_str"]) for m in metadata])
    is_female = np.array([labels[m["caption"]][0] for m in metadata])
    is_male = np.array([labels[m["caption"]][1] for m in metadata])
    return is_content, is_female, is_male


def base_rates(is_content: np.ndarray, is_female: np.ndarray, is_male: np.ndarray) -> BaseRates:
    """Fractions of content-word entries whose caption is female / male."""
    n_content = is_content.sum()
    return BaseRates(female=float((is_female & is_content).sum() / n_content),
                     male=float((is_male & is_content).sum() / n_content))


class LALSScorer:
    """Scores the visual tokens of an image against one layer of the text database
    (vectorised form of :func:`token_lals`).

        idx = scorer.neighbours(hidden)          # k nearest database entries of every visual token
        scorer.scores_from_neighbours(idx)       # LALS of every token
        scorer.token_scores(hidden)              # LALS of tokens 1..n-1 (layer sweeps)
        scorer.image_lals(idx)                   # image score used by the token maps / colour ablation
        scorer.token_map(idx, (rows, cols))      # token scores laid out on the image grid
    """

    def __init__(self, db: TextDB, k: int = 20, female_terms: frozenset[str] = FEMALE_TERMS,
                 male_terms: frozenset[str] = MALE_TERMS, backend: str = "exact", nprobe: int = 32) -> None:
        self.db = db
        self.k = k
        self.backend = backend
        self.nprobe = nprobe
        self.is_content, self.is_female, self.is_male = label_entries(db.metadata, female_terms, male_terms)
        self.base_rates = base_rates(self.is_content, self.is_female, self.is_male)

    def neighbours(self, hidden_states) -> np.ndarray:
        """k-NN indices ``[n_tokens, k]`` of visual-token hidden states ``[n_tokens, dim]``."""
        return self.db.search(hidden_states, self.k, backend=self.backend, nprobe=self.nprobe)

    def fractions(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per token: f, m and the number of content-word neighbours (f = m = 0 if there is none)."""
        content = self.is_content[idx]
        n_content = content.sum(axis=1)
        safe = np.maximum(n_content, 1)
        female = (self.is_female[idx] & content).sum(axis=1) / safe
        male = (self.is_male[idx] & content).sum(axis=1) / safe
        return female, male, n_content

    def scores_from_neighbours(self, idx: np.ndarray) -> np.ndarray:
        """LALS of every row of a ``[n_tokens, k]`` neighbour-index matrix."""
        female, male, n_content = self.fractions(idx)
        net = (female - self.base_rates.female) - (male - self.base_rates.male)
        return np.where(n_content > 0, net, 0.0)

    def token_scores(self, hidden_states) -> np.ndarray:
        """LALS of visual tokens 1..n-1 (token 0 excluded)."""
        return self.scores_from_neighbours(self.neighbours(hidden_states))[1:]

    def image_lals(self, idx: np.ndarray) -> float:
        """Image LALS ``(mean f - f_base) - (mean m - m_base)`` over tokens 1..n-1, as used by the
        token maps and the colour ablation. Same as the mean token score, except that tokens
        without content-word neighbours count as ``m_base - f_base`` (about 1e-4) instead of 0."""
        female, male, _ = self.fractions(idx[1:])
        return float((female.mean() - self.base_rates.female) - (male.mean() - self.base_rates.male))

    def token_map(self, idx: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
        """Token scores on the visual-token grid (row-major, ``grid = (rows, cols)``); NaN for token 0
        and for tokens without content-word neighbours."""
        female, male, n_content = self.fractions(idx)
        scores = (female - self.base_rates.female) - (male - self.base_rates.male)
        rows, cols = grid
        flat = np.full(rows * cols, np.nan)
        n = min(len(scores), rows * cols)
        flat[:n] = np.where(n_content[:n] > 0, scores[:n], np.nan)
        flat[0] = np.nan
        return flat.reshape(rows, cols)


def tail_means(scores: np.ndarray, pct: float = 5.0) -> tuple[float, float]:
    """Means of the most female (highest) and most male (lowest) ``pct``% of tokens."""
    n = max(1, int(len(scores) * pct / 100))
    ordered = np.sort(scores)
    return float(ordered[-n:].mean()), float(ordered[:n].mean())


def top_abs_mean(scores: np.ndarray, pct: float = 5.0) -> float:
    """Signed mean LALS of the ``pct``% of tokens with the largest ``|LALS|`` (paper Eq. 2)."""
    n = max(1, int(round(len(scores) * pct / 100.0)))
    return float(scores[np.argsort(np.abs(scores))[::-1][:n]].mean())


def image_summary(scores: np.ndarray) -> dict[str, float]:
    """Per-image record of the layer sweeps: mean over all tokens (plotted) and the 5% tails."""
    top5_female, top5_male = tail_means(scores, 5.0)
    return {"mean": float(scores.mean()), "top5_female": top5_female, "top5_male": top5_male}
