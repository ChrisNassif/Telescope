"""
AST-based Random Symbolic Function Generator and Evaluator for LLM Text Detectors.

Includes monotonic equivalence filtering to eliminate functions that produce identical
rank orderings under linear decision thresholding (e.g. exp(metric), scalar scaling,
or metrics with Spearman rank correlation |rho| > 0.999 to previously tested functions).
"""

import math
import random
import warnings
from typing import List, Dict, Any, Tuple, Set, Optional, Union
import numpy as np
from scipy.stats import spearmanr


# Supported 16 base per-token metric names
METRIC_NAMES: List[str] = [
    "log_prob",
    "shift1_log_prob",
    "shift2_log_prob",
    "shift3_log_prob",
    "rank",
    "shift1_rank",
    "shift2_rank",
    "shift3_rank",
    "log_rank",
    "shift1_log_rank",
    "shift2_log_rank",
    "shift3_log_rank",
    "entropy",
    "mean_ref",
    "var_ref",
    "max_prob",
    "top2_prob",
    "margin_prob",
    "target_logit_diff",
    "top5_mass",
    "top10_mass",
    "tv_dist",
    "cross_loss",
    "d_log_prob",
    "d_entropy",
]

# Supported unary operations at token level or sequence level
UNARY_OPS: List[str] = ["log", "exp", "sqrt", "square", "abs", "negate"]

# Supported binary operations
BINARY_OPS: List[str] = ["+", "-", "*", "/"]

# Supported sequence aggregators (reducing T -> 1 scalar per sample)
AGGREGATORS: List[str] = ["mean", "sum", "median", "std"]


class ASTNode:
    """Base class for AST expression nodes."""

    def evaluate_tokens(self, token_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluates expression at per-token level across samples."""
        raise NotImplementedError

    def evaluate_sequence(self, token_data: Dict[str, np.ndarray], sample_offsets: np.ndarray, sample_lengths: np.ndarray) -> np.ndarray:
        """Evaluates expression into a 1D scalar score array per sample."""
        raise NotImplementedError

    def to_string(self) -> str:
        """Returns string representation of formula."""
        raise NotImplementedError

    def canonicalize(self) -> "ASTNode":
        """Returns simplified canonical AST to eliminate outer monotonic transformations."""
        return self


class MetricNode(ASTNode):
    """Leaf node representing a base per-token metric."""

    def __init__(self, metric_name: str) -> None:
        if metric_name not in METRIC_NAMES:
            raise ValueError(f"Unknown metric name: {metric_name}")
        self.metric_name = metric_name

    def evaluate_tokens(self, token_data: Dict[str, np.ndarray]) -> np.ndarray:
        return token_data[self.metric_name]

    def to_string(self) -> str:
        return self.metric_name


class UnaryOpNode(ASTNode):
    """Node representing a unary operation (e.g. log, exp, sqrt, square, abs, negate)."""

    def __init__(self, op: str, child: ASTNode) -> None:
        if op not in UNARY_OPS:
            raise ValueError(f"Unknown unary op: {op}")
        self.op = op
        self.child = child

    def evaluate_tokens(self, token_data: Dict[str, np.ndarray]) -> np.ndarray:
        with np.errstate(all='ignore'):
            val: np.ndarray = self.child.evaluate_tokens(token_data)
            if self.op == "log":
                return np.log(np.maximum(val, 1e-10))
            elif self.op == "exp":
                val_clipped = np.clip(val, -50.0, 50.0)
                return np.exp(val_clipped)
            elif self.op == "sqrt":
                return np.sqrt(np.maximum(val, 0.0))
            elif self.op == "square":
                val_clipped = np.clip(val, -1e4, 1e4)
                return np.square(val_clipped)
            elif self.op == "abs":
                return np.abs(val)
            elif self.op == "negate":
                return -val
            else:
                raise ValueError(f"Unhandled unary op: {self.op}")

    def to_string(self) -> str:
        return f"{self.op}({self.child.to_string()})"

    def canonicalize(self) -> ASTNode:
        child_canon: ASTNode = self.child.canonicalize()
        # Exp, log, sqrt, abs, negate on outer level under linear thresholding are monotonic
        # e.g., exp(x) under rank thresholding produces identical ROC-AUC to x.
        if self.op in ("exp", "sqrt", "log"):
            return child_canon
        return UnaryOpNode(self.op, child_canon)


class BinaryOpNode(ASTNode):
    """Node representing a binary operation (+, -, *, /)."""

    def __init__(self, op: str, left: ASTNode, right: ASTNode) -> None:
        if op not in BINARY_OPS:
            raise ValueError(f"Unknown binary op: {op}")
        self.op = op
        self.left = left
        self.right = right

    def evaluate_tokens(self, token_data: Dict[str, np.ndarray]) -> np.ndarray:
        with np.errstate(all='ignore'):
            left_val: np.ndarray = self.left.evaluate_tokens(token_data)
            right_val: np.ndarray = self.right.evaluate_tokens(token_data)

            if self.op == "+":
                return np.nan_to_num(left_val + right_val, nan=0.0, posinf=1e5, neginf=-1e5)
            elif self.op == "-":
                return np.nan_to_num(left_val - right_val, nan=0.0, posinf=1e5, neginf=-1e5)
            elif self.op == "*":
                return np.nan_to_num(left_val * right_val, nan=0.0, posinf=1e5, neginf=-1e5)
            elif self.op == "/":
                return np.nan_to_num(left_val / (right_val + np.sign(right_val + 1e-10) * 1e-10), nan=0.0, posinf=1e5, neginf=-1e5)
            else:
                raise ValueError(f"Unhandled binary op: {self.op}")

    def to_string(self) -> str:
        left_str: str = self.left.to_string()
        right_str: str = self.right.to_string()
        return f"({left_str} {self.op} {right_str})"

    def canonicalize(self) -> ASTNode:
        left_canon: ASTNode = self.left.canonicalize()
        right_canon: ASTNode = self.right.canonicalize()

        # Commutative sorting for + and * to deduplicate equivalent ordering
        if self.op in ("+", "*") and left_canon.to_string() > right_canon.to_string():
            left_canon, right_canon = right_canon, left_canon

        return BinaryOpNode(self.op, left_canon, right_canon)


class AggregateNode(ASTNode):
    """Top-level node that aggregates per-token streams into a scalar per sample."""

    def __init__(self, aggregator: str, child: ASTNode) -> None:
        if aggregator not in AGGREGATORS:
            raise ValueError(f"Unknown aggregator: {aggregator}. Must be one of {AGGREGATORS}")
        self.aggregator = aggregator
        self.child = child

    def evaluate_tokens(self, token_data: Dict[str, np.ndarray]) -> np.ndarray:
        return self.child.evaluate_tokens(token_data)

    def evaluate_sequence(self, token_data: Dict[str, np.ndarray], sample_offsets: np.ndarray, sample_lengths: np.ndarray) -> np.ndarray:
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            token_vals: np.ndarray = self.child.evaluate_tokens(token_data)

            # Handle NaNs / Inf gracefully
            token_vals = np.nan_to_num(token_vals, nan=0.0, posinf=1e5, neginf=-1e5)

            num_samples: int = len(sample_offsets)
            if num_samples == 0 or len(token_vals) == 0:
                return np.zeros(num_samples, dtype=np.float32)

            if self.aggregator == "sum":
                scores = np.add.reduceat(token_vals, sample_offsets)
            elif self.aggregator == "mean":
                sums = np.add.reduceat(token_vals, sample_offsets)
                scores = sums / np.maximum(sample_lengths, 1)
            elif self.aggregator == "median":
                scores = np.zeros(num_samples, dtype=np.float32)
                for idx in range(num_samples):
                    st, l = sample_offsets[idx], sample_lengths[idx]
                    if l > 0:
                        scores[idx] = np.median(token_vals[st : st + l])
            elif self.aggregator == "std":
                means = np.add.reduceat(token_vals, sample_offsets) / np.maximum(sample_lengths, 1)
                means_expanded = np.repeat(means, sample_lengths)
                sq_diff = np.square(token_vals[:len(means_expanded)] - means_expanded)
                scores = np.sqrt(np.add.reduceat(sq_diff, sample_offsets) / np.maximum(sample_lengths, 1))
            else:
                raise ValueError(f"Unknown aggregator: {self.aggregator}")

            return np.nan_to_num(scores, nan=0.0, posinf=1e5, neginf=-1e5)

    def to_string(self) -> str:
        return f"{self.aggregator}({self.child.to_string()})"

    def canonicalize(self) -> ASTNode:
        child_canon: ASTNode = self.child.canonicalize()
        return AggregateNode(self.aggregator, child_canon)


def generate_random_ast(max_depth: int = 3, current_depth: int = 0) -> ASTNode:
    """Generates a random token-level AST expression."""
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.35):
        return MetricNode(random.choice(METRIC_NAMES))

    choice: float = random.random()
    if choice < 0.5:
        # Unary operation
        op: str = random.choice(UNARY_OPS)
        child: ASTNode = generate_random_ast(max_depth, current_depth + 1)
        return UnaryOpNode(op, child)
    else:
        # Binary operation
        op: str = random.choice(BINARY_OPS)
        left: ASTNode = generate_random_ast(max_depth, current_depth + 1)
        right: ASTNode = generate_random_ast(max_depth, current_depth + 1)
        return BinaryOpNode(op, left, right)


def generate_random_detector(max_depth: int = 3) -> AggregateNode:
    """Generates a full random detector expression (aggregator over token AST)."""
    aggregator: str = random.choice(AGGREGATORS)
    token_ast: ASTNode = generate_random_ast(max_depth=max_depth, current_depth=0)
    return AggregateNode(aggregator, token_ast)


class NonRedundantDetectorPool:
    """
    Pool that manages candidate detector expressions and filters out redundant expressions.

    Uses both:
    1. Canonical string deduplication (outer monotonic stripping, commutative sorting).
    2. Empirical rank correlation check (Spearman |rho| > 0.999 on validation batch).
    """

    def __init__(self, rho_threshold: float = 0.999) -> None:
        self.rho_threshold = rho_threshold
        self.canonical_strings: Set[str] = set()
        self.rank_matrix: Optional[np.ndarray] = None  # Shape: (num_detectors, num_samples)
        self.valid_detectors: List[AggregateNode] = []

    def _normalize_rank(self, scores: np.ndarray) -> Optional[np.ndarray]:
        if np.all(np.isnan(scores)) or np.std(scores) < 1e-9:
            return None
        ranks = np.argsort(np.argsort(scores)).astype(np.float32)
        std_r = np.std(ranks)
        if std_r < 1e-9:
            return None
        return (ranks - np.mean(ranks)) / std_r

    def is_redundant(
        self,
        detector: AggregateNode,
        validation_token_data: Optional[Dict[str, np.ndarray]] = None,
        sample_offsets: Optional[np.ndarray] = None,
        sample_lengths: Optional[np.ndarray] = None,
    ) -> bool:
        canon_str: str = detector.canonicalize().to_string()
        if canon_str in self.canonical_strings:
            return True

        if validation_token_data is not None and sample_offsets is not None and sample_lengths is not None and self.rank_matrix is not None:
            try:
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores: np.ndarray = detector.evaluate_sequence(validation_token_data, sample_offsets, sample_lengths)
                    norm_r = self._normalize_rank(scores)
                    if norm_r is None:
                        return True

                    rhos = np.abs(np.dot(self.rank_matrix, norm_r) / len(norm_r))
                    if np.any(rhos >= self.rho_threshold):
                        return True
            except Exception:
                return True

        return False

    def add_detector(self, detector: AggregateNode, scores: Optional[np.ndarray] = None) -> bool:
        canon_str: str = detector.canonicalize().to_string()
        self.canonical_strings.add(canon_str)
        self.valid_detectors.append(detector)
        if scores is not None:
            norm_r = self._normalize_rank(scores)
            if norm_r is not None:
                if self.rank_matrix is None:
                    self.rank_matrix = norm_r.reshape(1, -1)
                else:
                    self.rank_matrix = np.vstack([self.rank_matrix, norm_r])
        return True


def generate_unique_detectors(
    num_detectors: int,
    validation_token_data: Optional[Dict[str, np.ndarray]] = None,
    sample_offsets: Optional[np.ndarray] = None,
    sample_lengths: Optional[np.ndarray] = None,
    max_depth: int = 3,
    max_attempts: int = 50000,
) -> List[AggregateNode]:
    """Generates a list of unique, non-redundant detector expressions."""
    pool: NonRedundantDetectorPool = NonRedundantDetectorPool()
    attempts: int = 0
    consecutive_failures: int = 0

    while len(pool.valid_detectors) < num_detectors and attempts < max_attempts and consecutive_failures < 2000:
        attempts += 1
        cand: AggregateNode = generate_random_detector(max_depth=max_depth)

        if pool.is_redundant(cand, validation_token_data, sample_offsets, sample_lengths):
            consecutive_failures += 1
            continue

        consecutive_failures = 0
        scores: Optional[np.ndarray] = None
        if validation_token_data is not None and sample_offsets is not None and sample_lengths is not None:
            scores = cand.evaluate_sequence(validation_token_data, sample_offsets, sample_lengths)

        pool.add_detector(cand, scores)

    print(f"Generated {len(pool.valid_detectors)} non-redundant detectors after {attempts} attempts.")
    return pool.valid_detectors
