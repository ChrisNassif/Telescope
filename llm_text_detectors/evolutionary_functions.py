"""
Genetic Programming & Evolutionary Search Engine for LLM Text Detectors.

Evolves symbolic AST expressions by using high-fitness detectors as parent bases
for crossover (subtree exchange) and mutation, prioritizing successful building blocks.
"""

import copy
import random
import warnings
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

from llm_text_detectors.random_functions import (
    ASTNode,
    MetricNode,
    UnaryOpNode,
    BinaryOpNode,
    AggregateNode,
    NonRedundantDetectorPool,
    generate_random_ast,
    generate_random_detector,
    METRIC_NAMES,
    UNARY_OPS,
    BINARY_OPS,
    AGGREGATORS,
)


def clone_ast(node: ASTNode) -> ASTNode:
    """Creates a deep copy of an AST node tree."""
    return copy.deepcopy(node)


def collect_token_subtrees(node: ASTNode) -> List[ASTNode]:
    """Collects all token-level AST subnodes (excluding top AggregateNode)."""
    subnodes: List[ASTNode] = [node]
    if isinstance(node, UnaryOpNode):
        subnodes.extend(collect_token_subtrees(node.child))
    elif isinstance(node, BinaryOpNode):
        subnodes.extend(collect_token_subtrees(node.left))
        subnodes.extend(collect_token_subtrees(node.right))
    return subnodes


def replace_token_subtree(root: ASTNode, target: ASTNode, replacement: ASTNode) -> ASTNode:
    """Replaces a target subnode in the AST with a replacement subnode."""
    if root is target:
        return clone_ast(replacement)

    if isinstance(root, UnaryOpNode):
        new_child = replace_token_subtree(root.child, target, replacement)
        return UnaryOpNode(root.op, new_child)
    elif isinstance(root, BinaryOpNode):
        new_left = replace_token_subtree(root.left, target, replacement)
        new_right = replace_token_subtree(root.right, target, replacement)
        return BinaryOpNode(root.op, new_left, new_right)
    elif isinstance(root, AggregateNode):
        new_child = replace_token_subtree(root.child, target, replacement)
        return AggregateNode(root.aggregator, new_child)
    else:
        return root


def mutate_detector(detector: AggregateNode, mutation_rate: float = 0.3, max_depth: int = 3) -> AggregateNode:
    """Mutates a detector expression (modifying leaf metrics, operators, or subtrees)."""
    detector_copy: AggregateNode = clone_ast(detector)

    # Optionally mutate top-level aggregator
    if random.random() < 0.15:
        detector_copy.aggregator = random.choice(AGGREGATORS)

    subnodes: List[ASTNode] = collect_token_subtrees(detector_copy.child)
    if not subnodes:
        return detector_copy

    target_node: ASTNode = random.choice(subnodes)

    mutation_type: float = random.random()
    if mutation_type < 0.4:
        # Replace target subtree with a brand new random sub-AST
        new_sub: ASTNode = generate_random_ast(max_depth=max_depth)
        detector_copy.child = replace_token_subtree(detector_copy.child, target_node, new_sub)

    elif mutation_type < 0.7:
        # Change operator / metric in place
        if isinstance(target_node, MetricNode):
            new_metric = random.choice([m for m in METRIC_NAMES if m != target_node.metric_name])
            replacement = MetricNode(new_metric)
            detector_copy.child = replace_token_subtree(detector_copy.child, target_node, replacement)
        elif isinstance(target_node, UnaryOpNode):
            new_op = random.choice([o for o in UNARY_OPS if o != target_node.op])
            replacement = UnaryOpNode(new_op, clone_ast(target_node.child))
            detector_copy.child = replace_token_subtree(detector_copy.child, target_node, replacement)
        elif isinstance(target_node, BinaryOpNode):
            new_op = random.choice([o for o in BINARY_OPS if o != target_node.op])
            replacement = BinaryOpNode(new_op, clone_ast(target_node.left), clone_ast(target_node.right))
            detector_copy.child = replace_token_subtree(detector_copy.child, target_node, replacement)

    else:
        # Wrap target node in a new unary op or binary op
        op_wrap = random.choice(UNARY_OPS)
        wrapped = UnaryOpNode(op_wrap, clone_ast(target_node))
        detector_copy.child = replace_token_subtree(detector_copy.child, target_node, wrapped)

    return detector_copy


def crossover_detectors(parent1: AggregateNode, parent2: AggregateNode) -> AggregateNode:
    """
    Combines successful building blocks from parent1 (donor) into parent2 (recipient).
    A random subtree from parent1 is inserted into parent2.
    """
    subnodes_donor: List[ASTNode] = collect_token_subtrees(parent1.child)
    subnodes_recipient: List[ASTNode] = collect_token_subtrees(parent2.child)

    if not subnodes_donor or not subnodes_recipient:
        return clone_ast(parent1)

    donor_sub: ASTNode = random.choice(subnodes_donor)
    recipient_sub: ASTNode = random.choice(subnodes_recipient)

    child_copy: AggregateNode = clone_ast(parent2)
    child_copy.child = replace_token_subtree(child_copy.child, recipient_sub, donor_sub)
    return child_copy


def tournament_selection(population: List[AggregateNode], fitness_scores: List[float], tournament_size: int = 3) -> AggregateNode:
    """Selects a parent using tournament selection (favoring higher fitness)."""
    selected_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    best_idx = max(selected_indices, key=lambda i: fitness_scores[i])
    return population[best_idx]
