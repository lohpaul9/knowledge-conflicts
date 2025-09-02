#!/usr/bin/env python3
"""
Analyze causality results for a fixed layer (18) to find optimal intervention strength.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CausalityResult:
    """Causality experiment result."""

    layer_name: str
    prompt_type: str
    experiment_type: str
    intervention_type: str
    intervention_strength: float
    original_logits: np.ndarray
    modified_logits: np.ndarray
    original_probs: np.ndarray
    modified_probs: np.ndarray
    entropy_change: float
    unsure_prob_change: float
    answer_token_probs: Dict[str, float]


def load_results(pickle_path: str) -> Tuple[List[CausalityResult], Dict]:
    """Load results from pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old and new pickle formats
    if isinstance(data, dict):
        if "results" in data:
            results = data["results"]
            analysis = data.get("analysis", {})
        else:
            # Old format where data is directly the results
            results = data
            analysis = {}
    else:
        results = data
        analysis = {}

    return results, analysis


def filter_layer_18_results(results: List[CausalityResult]) -> List[CausalityResult]:
    """Filter results to only include layer 18."""
    return [r for r in results if "blocks.18.hook_resid_post" in r.layer_name]


def analyze_intervention_strengths(results: List[CausalityResult]) -> Dict:
    """Analyze the impact of different intervention strengths."""

    # Group by intervention type and strength
    strength_analysis = {}

    for intervention_type in ["add", "subtract"]:
        strength_analysis[intervention_type] = {}

        for result in results:
            if result.intervention_type != intervention_type:
                continue

            strength = result.intervention_strength

            if strength not in strength_analysis[intervention_type]:
                strength_analysis[intervention_type][strength] = {
                    "entropy_changes": [],
                    "unsure_changes": [],
                    "count": 0,
                }

            strength_analysis[intervention_type][strength]["entropy_changes"].append(
                result.entropy_change
            )
            strength_analysis[intervention_type][strength]["unsure_changes"].append(
                result.unsure_prob_change
            )
            strength_analysis[intervention_type][strength]["count"] += 1

    # Compute statistics
    for intervention_type in strength_analysis:
        for strength in strength_analysis[intervention_type]:
            data = strength_analysis[intervention_type][strength]
            data["mean_entropy_change"] = np.mean(data["entropy_changes"])
            data["std_entropy_change"] = np.std(data["entropy_changes"])
            data["mean_unsure_change"] = np.mean(data["unsure_changes"])
            data["std_unsure_change"] = np.std(data["unsure_changes"])
            data["abs_mean_entropy_change"] = abs(data["mean_entropy_change"])
            data["abs_mean_unsure_change"] = abs(data["mean_unsure_change"])

    return strength_analysis


def find_optimal_strength(strength_analysis: Dict) -> Tuple[str, float, Dict]:
    """Find the intervention type and strength with the greatest impact."""

    max_entropy_impact = 0
    max_unsure_impact = 0
    optimal_entropy_config = None
    optimal_unsure_config = None

    for intervention_type in strength_analysis:
        for strength, data in strength_analysis[intervention_type].items():
            # Check entropy impact
            if data["abs_mean_entropy_change"] > max_entropy_impact:
                max_entropy_impact = data["abs_mean_entropy_change"]
                optimal_entropy_config = {
                    "intervention_type": intervention_type,
                    "strength": strength,
                    "mean_entropy_change": data["mean_entropy_change"],
                    "std_entropy_change": data["std_entropy_change"],
                }

            # Check unsure impact
            if data["abs_mean_unsure_change"] > max_unsure_impact:
                max_unsure_impact = data["abs_mean_unsure_change"]
                optimal_unsure_config = {
                    "intervention_type": intervention_type,
                    "strength": strength,
                    "mean_unsure_change": data["mean_unsure_change"],
                    "std_unsure_change": data["std_unsure_change"],
                }

    return optimal_entropy_config, optimal_unsure_config


def plot_strength_analysis(strength_analysis: Dict, save_path: str = None):
    """Plot the impact of different intervention strengths."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data for plotting
    add_strengths = sorted(strength_analysis["add"].keys())
    subtract_strengths = sorted(strength_analysis["subtract"].keys())

    # Plot 1: Entropy changes by strength (add)
    add_entropy_means = [
        strength_analysis["add"][s]["mean_entropy_change"] for s in add_strengths
    ]
    add_entropy_stds = [
        strength_analysis["add"][s]["std_entropy_change"] for s in add_strengths
    ]

    ax1.errorbar(
        add_strengths,
        add_entropy_means,
        yerr=add_entropy_stds,
        marker="o",
        color="red",
        label="Add",
        capsize=5,
    )
    ax1.set_xlabel("Intervention Strength")
    ax1.set_ylabel("Mean Entropy Change")
    ax1.set_title("Entropy Change: Adding Conflict Direction")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="--")
    ax1.legend()

    # Plot 2: Entropy changes by strength (subtract)
    subtract_entropy_means = [
        strength_analysis["subtract"][s]["mean_entropy_change"]
        for s in subtract_strengths
    ]
    subtract_entropy_stds = [
        strength_analysis["subtract"][s]["std_entropy_change"]
        for s in subtract_strengths
    ]

    ax2.errorbar(
        subtract_strengths,
        subtract_entropy_means,
        yerr=subtract_entropy_stds,
        marker="o",
        color="blue",
        label="Subtract",
        capsize=5,
    )
    ax2.set_xlabel("Intervention Strength")
    ax2.set_ylabel("Mean Entropy Change")
    ax2.set_title("Entropy Change: Subtracting Conflict Direction")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="--")
    ax2.legend()

    # Plot 3: Unsure changes by strength (add)
    add_unsure_means = [
        strength_analysis["add"][s]["mean_unsure_change"] for s in add_strengths
    ]
    add_unsure_stds = [
        strength_analysis["add"][s]["std_unsure_change"] for s in add_strengths
    ]

    ax3.errorbar(
        add_strengths,
        add_unsure_means,
        yerr=add_unsure_stds,
        marker="o",
        color="orange",
        label="Add",
        capsize=5,
    )
    ax3.set_xlabel("Intervention Strength")
    ax3.set_ylabel("Mean Unsure Probability Change")
    ax3.set_title("Unsure Probability Change: Adding Conflict Direction")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="black", linestyle="--")
    ax3.legend()

    # Plot 4: Unsure changes by strength (subtract)
    subtract_unsure_means = [
        strength_analysis["subtract"][s]["mean_unsure_change"]
        for s in subtract_strengths
    ]
    subtract_unsure_stds = [
        strength_analysis["subtract"][s]["std_unsure_change"]
        for s in subtract_strengths
    ]

    ax4.errorbar(
        subtract_strengths,
        subtract_unsure_means,
        yerr=subtract_unsure_stds,
        marker="o",
        color="green",
        label="Subtract",
        capsize=5,
    )
    ax4.set_xlabel("Intervention Strength")
    ax4.set_ylabel("Mean Unsure Probability Change")
    ax4.set_title("Unsure Probability Change: Subtracting Conflict Direction")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="black", linestyle="--")
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Layer 18 strength analysis plot saved to {save_path}")
    else:
        plt.show()


def analyze_optimal_config(
    results: List[CausalityResult], optimal_config: Dict
) -> Dict:
    """Analyze the impact of the optimal configuration."""

    # Filter results for the optimal configuration
    optimal_results = [
        r
        for r in results
        if (
            r.intervention_type == optimal_config["intervention_type"]
            and r.intervention_strength == optimal_config["strength"]
        )
    ]

    # Group by prompt type and experiment type
    analysis = {}

    for prompt_type in ["clean", "conflict"]:
        analysis[prompt_type] = {}
        for experiment_type in ["entropy", "unsure"]:
            filtered_results = [
                r
                for r in optimal_results
                if r.prompt_type == prompt_type and r.experiment_type == experiment_type
            ]

            if filtered_results:
                analysis[prompt_type][experiment_type] = {
                    "mean_entropy_change": np.mean(
                        [r.entropy_change for r in filtered_results]
                    ),
                    "std_entropy_change": np.std(
                        [r.entropy_change for r in filtered_results]
                    ),
                    "mean_unsure_change": np.mean(
                        [r.unsure_prob_change for r in filtered_results]
                    ),
                    "std_unsure_change": np.std(
                        [r.unsure_prob_change for r in filtered_results]
                    ),
                    "count": len(filtered_results),
                }

    return analysis


def print_analysis_summary(
    strength_analysis: Dict,
    optimal_entropy_config: Dict,
    optimal_unsure_config: Dict,
    optimal_analysis: Dict,
):
    """Print a summary of the analysis."""

    print("\n" + "=" * 60)
    print("LAYER 18 INTERVENTION STRENGTH ANALYSIS")
    print("=" * 60)

    print("\n1. INTERVENTION STRENGTH IMPACT:")
    print("-" * 40)

    for intervention_type in ["add", "subtract"]:
        print(f"\n{intervention_type.upper()} INTERVENTION:")
        for strength in sorted(strength_analysis[intervention_type].keys()):
            data = strength_analysis[intervention_type][strength]
            print(f"  Strength {strength}:")
            print(
                f"    Entropy change: {data['mean_entropy_change']:.4f} ± {data['std_entropy_change']:.4f}"
            )
            print(
                f"    Unsure change: {data['mean_unsure_change']:.4f} ± {data['std_unsure_change']:.4f}"
            )
            print(f"    Sample count: {data['count']}")

    print("\n2. OPTIMAL CONFIGURATIONS:")
    print("-" * 40)

    if optimal_entropy_config:
        print(f"\nGREATEST ENTROPY IMPACT:")
        print(f"  Intervention: {optimal_entropy_config['intervention_type']}")
        print(f"  Strength: {optimal_entropy_config['strength']}")
        print(
            f"  Mean entropy change: {optimal_entropy_config['mean_entropy_change']:.4f} ± {optimal_entropy_config['std_entropy_change']:.4f}"
        )

    if optimal_unsure_config:
        print(f"\nGREATEST UNSURE IMPACT:")
        print(f"  Intervention: {optimal_unsure_config['intervention_type']}")
        print(f"  Strength: {optimal_unsure_config['strength']}")
        print(
            f"  Mean unsure change: {optimal_unsure_config['mean_unsure_change']:.4f} ± {optimal_unsure_config['std_unsure_change']:.4f}"
        )

    print("\n3. OPTIMAL CONFIGURATION DETAILED ANALYSIS:")
    print("-" * 40)

    for prompt_type in ["clean", "conflict"]:
        print(f"\n{prompt_type.upper()} PROMPTS:")
        for experiment_type in ["entropy", "unsure"]:
            if experiment_type in optimal_analysis.get(prompt_type, {}):
                data = optimal_analysis[prompt_type][experiment_type]
                print(f"  {experiment_type.upper()} experiments:")
                print(
                    f"    Entropy change: {data['mean_entropy_change']:.4f} ± {data['std_entropy_change']:.4f}"
                )
                print(
                    f"    Unsure change: {data['mean_unsure_change']:.4f} ± {data['std_unsure_change']:.4f}"
                )
                print(f"    Sample count: {data['count']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze layer 18 causality results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to causality results pickle file",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="layer_18_analysis.png",
        help="Path to save analysis plot",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="blocks.18.hook_resid_post",
        help="Layer name to analyze",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results}...")
    results, _ = load_results(args.results)

    # Filter for layer 18
    print(f"Filtering for layer: {args.layer}")
    layer_results = [r for r in results if args.layer in r.layer_name]

    if not layer_results:
        print(f"No results found for layer {args.layer}")
        return

    print(f"Found {len(layer_results)} results for layer {args.layer}")

    # Analyze intervention strengths
    print("Analyzing intervention strengths...")
    strength_analysis = analyze_intervention_strengths(layer_results)

    # Find optimal configurations
    optimal_entropy_config, optimal_unsure_config = find_optimal_strength(
        strength_analysis
    )

    # Analyze optimal configuration (use entropy config as default)
    optimal_config = (
        optimal_entropy_config if optimal_entropy_config else optimal_unsure_config
    )
    if optimal_config:
        optimal_analysis = analyze_optimal_config(layer_results, optimal_config)
    else:
        optimal_analysis = {}

    # Print summary
    print_analysis_summary(
        strength_analysis,
        optimal_entropy_config,
        optimal_unsure_config,
        optimal_analysis,
    )

    # Create plot
    print(f"Creating plot...")
    plot_strength_analysis(strength_analysis, save_path=args.plot)


if __name__ == "__main__":
    main()
