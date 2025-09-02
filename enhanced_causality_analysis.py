#!/usr/bin/env python3
"""
Enhanced causality analysis that generates plots for both adding and subtracting
from clean and conflicting prompts using test_causality_results_32.pkl
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dataclasses import dataclass
import argparse


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


def analyze_prompt_intervention_effects(results: List[CausalityResult]) -> Dict:
    """Analyze the effects of adding/subtracting on clean vs conflict prompts."""

    analysis = {}

    for prompt_type in ["clean", "conflict"]:
        analysis[prompt_type] = {}

        for intervention_type in ["add", "subtract"]:
            analysis[prompt_type][intervention_type] = {}

            # Filter results for this combination
            filtered_results = [
                r
                for r in results
                if r.prompt_type == prompt_type
                and r.intervention_type == intervention_type
            ]

            if not filtered_results:
                continue

            # Group by strength
            for strength in sorted(
                set(r.intervention_strength for r in filtered_results)
            ):
                strength_results = [
                    r for r in filtered_results if r.intervention_strength == strength
                ]

                analysis[prompt_type][intervention_type][strength] = {
                    "entropy_changes": [r.entropy_change for r in strength_results],
                    "unsure_changes": [r.unsure_prob_change for r in strength_results],
                    "count": len(strength_results),
                    "mean_entropy_change": np.mean(
                        [r.entropy_change for r in strength_results]
                    ),
                    "std_entropy_change": np.std(
                        [r.entropy_change for r in strength_results]
                    ),
                    "mean_unsure_change": np.mean(
                        [r.unsure_prob_change for r in strength_results]
                    ),
                    "std_unsure_change": np.std(
                        [r.unsure_prob_change for r in strength_results]
                    ),
                }

    return analysis


def plot_prompt_intervention_analysis(analysis: Dict, save_path: str = None):
    """Create comprehensive plots for prompt-intervention analysis."""

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))

    # Create a grid layout with more vertical spacing
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.3)

    # Colors for different intervention types
    colors = {"add": "red", "subtract": "blue"}
    markers = {"add": "o", "subtract": "s"}

    # Plot 1: Entropy changes for clean prompts (add vs subtract)
    ax1 = fig.add_subplot(gs[0, 0])
    for intervention_type in ["add", "subtract"]:
        if intervention_type in analysis["clean"]:
            strengths = sorted(analysis["clean"][intervention_type].keys())
            means = [
                analysis["clean"][intervention_type][s]["mean_entropy_change"]
                for s in strengths
            ]
            stds = [
                analysis["clean"][intervention_type][s]["std_entropy_change"]
                for s in strengths
            ]

            ax1.plot(
                strengths,
                means,
                marker=markers[intervention_type],
                color=colors[intervention_type],
                label=f"{intervention_type.capitalize()}",
                linewidth=2,
                markersize=8,
            )

    ax1.set_xlabel("Intervention Strength")
    ax1.set_ylabel("Mean Entropy Change")
    ax1.set_title("Clean Prompts: Entropy Changes", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.7)
    ax1.legend()
    ax1.set_xscale("log")

    # Plot 2: Entropy changes for conflict prompts (add vs subtract)
    ax2 = fig.add_subplot(gs[0, 1])
    for intervention_type in ["add", "subtract"]:
        if intervention_type in analysis["conflict"]:
            strengths = sorted(analysis["conflict"][intervention_type].keys())
            means = [
                analysis["conflict"][intervention_type][s]["mean_entropy_change"]
                for s in strengths
            ]
            stds = [
                analysis["conflict"][intervention_type][s]["std_entropy_change"]
                for s in strengths
            ]

            ax2.plot(
                strengths,
                means,
                marker=markers[intervention_type],
                color=colors[intervention_type],
                label=f"{intervention_type.capitalize()}",
                linewidth=2,
                markersize=8,
            )

    ax2.set_xlabel("Intervention Strength")
    ax2.set_ylabel("Mean Entropy Change")
    ax2.set_title("Conflict Prompts: Entropy Changes", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.7)
    ax2.legend()
    ax2.set_xscale("log")

    # Plot 3: Unsure changes for clean prompts (add vs subtract)
    ax3 = fig.add_subplot(gs[1, 0])
    for intervention_type in ["add", "subtract"]:
        if intervention_type in analysis["clean"]:
            strengths = sorted(analysis["clean"][intervention_type].keys())
            means = [
                analysis["clean"][intervention_type][s]["mean_unsure_change"]
                for s in strengths
            ]
            stds = [
                analysis["clean"][intervention_type][s]["std_unsure_change"]
                for s in strengths
            ]

            ax3.plot(
                strengths,
                means,
                marker=markers[intervention_type],
                color=colors[intervention_type],
                label=f"{intervention_type.capitalize()}",
                linewidth=2,
                markersize=8,
            )

    ax3.set_xlabel("Intervention Strength")
    ax3.set_ylabel("Mean Unsure Probability Change")
    ax3.set_title("Clean Prompts: Unsure Changes", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.7)
    ax3.legend()
    ax3.set_xscale("log")

    # Plot 4: Unsure changes for conflict prompts (add vs subtract)
    ax4 = fig.add_subplot(gs[1, 1])
    for intervention_type in ["add", "subtract"]:
        if intervention_type in analysis["conflict"]:
            strengths = sorted(analysis["conflict"][intervention_type].keys())
            means = [
                analysis["conflict"][intervention_type][s]["mean_unsure_change"]
                for s in strengths
            ]
            stds = [
                analysis["conflict"][intervention_type][s]["std_unsure_change"]
                for s in strengths
            ]

            ax4.plot(
                strengths,
                means,
                marker=markers[intervention_type],
                color=colors[intervention_type],
                label=f"{intervention_type.capitalize()}",
                linewidth=2,
                markersize=8,
            )

    ax4.set_xlabel("Intervention Strength")
    ax4.set_ylabel("Mean Unsure Probability Change")
    ax4.set_title("Conflict Prompts: Unsure Changes", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="black", linestyle="--", alpha=0.7)
    ax4.legend()
    ax4.set_xscale("log")

    # Plot 5: Comparison of clean vs conflict for adding (entropy)
    ax5 = fig.add_subplot(gs[2, 0])
    for prompt_type in ["clean", "conflict"]:
        if "add" in analysis[prompt_type]:
            strengths = sorted(analysis[prompt_type]["add"].keys())
            means = [
                analysis[prompt_type]["add"][s]["mean_entropy_change"]
                for s in strengths
            ]
            stds = [
                analysis[prompt_type]["add"][s]["std_entropy_change"] for s in strengths
            ]

            color = "green" if prompt_type == "clean" else "orange"
            marker = "o" if prompt_type == "clean" else "s"

            ax5.plot(
                strengths,
                means,
                marker=marker,
                color=color,
                label=f"{prompt_type.capitalize()}",
                linewidth=2,
                markersize=8,
            )

    ax5.set_xlabel("Intervention Strength")
    ax5.set_ylabel("Mean Entropy Change")
    ax5.set_title("Adding: Clean vs Conflict (Entropy)", fontsize=14, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color="black", linestyle="--", alpha=0.7)
    ax5.legend()
    ax5.set_xscale("log")

    # Plot 6: Comparison of clean vs conflict for subtracting (entropy)
    ax6 = fig.add_subplot(gs[2, 1])
    for prompt_type in ["clean", "conflict"]:
        if "subtract" in analysis[prompt_type]:
            strengths = sorted(analysis[prompt_type]["subtract"].keys())
            means = [
                analysis[prompt_type]["subtract"][s]["mean_entropy_change"]
                for s in strengths
            ]
            stds = [
                analysis[prompt_type]["subtract"][s]["std_entropy_change"]
                for s in strengths
            ]

            color = "green" if prompt_type == "clean" else "orange"
            marker = "o" if prompt_type == "clean" else "s"

            ax6.plot(
                strengths,
                means,
                marker=marker,
                color=color,
                label=f"{prompt_type.capitalize()}",
                linewidth=2,
                markersize=8,
            )

    ax6.set_xlabel("Intervention Strength")
    ax6.set_ylabel("Mean Entropy Change")
    ax6.set_title(
        "Subtracting: Clean vs Conflict (Entropy)", fontsize=14, fontweight="bold"
    )
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color="black", linestyle="--", alpha=0.7)
    ax6.legend()
    ax6.set_xscale("log")

    # Plot 7: Comparison of clean vs conflict for adding (unsure)
    ax7 = fig.add_subplot(gs[3, 0])
    for prompt_type in ["clean", "conflict"]:
        if "add" in analysis[prompt_type]:
            strengths = sorted(analysis[prompt_type]["add"].keys())
            means = [
                analysis[prompt_type]["add"][s]["mean_unsure_change"] for s in strengths
            ]
            stds = [
                analysis[prompt_type]["add"][s]["std_unsure_change"] for s in strengths
            ]

            color = "green" if prompt_type == "clean" else "orange"
            marker = "o" if prompt_type == "clean" else "s"

            ax7.plot(
                strengths,
                means,
                marker=marker,
                color=color,
                label=f"{prompt_type.capitalize()}",
                linewidth=2,
                markersize=8,
            )

    ax7.set_xlabel("Intervention Strength")
    ax7.set_ylabel("Mean Unsure Probability Change")
    ax7.set_title("Adding: Clean vs Conflict (Unsure)", fontsize=14, fontweight="bold")
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color="black", linestyle="--", alpha=0.7)
    ax7.legend()
    ax7.set_xscale("log")

    # Plot 8: Comparison of clean vs conflict for subtracting (unsure)
    ax8 = fig.add_subplot(gs[3, 1])
    for prompt_type in ["clean", "conflict"]:
        if "subtract" in analysis[prompt_type]:
            strengths = sorted(analysis[prompt_type]["subtract"].keys())
            means = [
                analysis[prompt_type]["subtract"][s]["mean_unsure_change"]
                for s in strengths
            ]
            stds = [
                analysis[prompt_type]["subtract"][s]["std_unsure_change"]
                for s in strengths
            ]

            color = "green" if prompt_type == "clean" else "orange"
            marker = "o" if prompt_type == "clean" else "s"

            ax8.plot(
                strengths,
                means,
                marker=marker,
                color=color,
                label=f"{prompt_type.capitalize()}",
                linewidth=2,
                markersize=8,
            )

    ax8.set_xlabel("Intervention Strength")
    ax8.set_ylabel("Mean Unsure Probability Change")
    ax8.set_title(
        "Subtracting: Clean vs Conflict (Unsure)", fontsize=14, fontweight="bold"
    )
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=0, color="black", linestyle="--", alpha=0.7)
    ax8.legend()
    ax8.set_xscale("log")

    # Add summary statistics as text
    ax_text = fig.add_subplot(gs[:, 2:])
    ax_text.axis("off")

    # Calculate summary statistics
    summary_text = "SUMMARY STATISTICS\n" + "=" * 50 + "\n\n"

    for prompt_type in ["clean", "conflict"]:
        summary_text += f"{prompt_type.upper()} PROMPTS:\n"
        summary_text += "-" * 30 + "\n"

        for intervention_type in ["add", "subtract"]:
            if intervention_type in analysis[prompt_type]:
                summary_text += f"\n{intervention_type.upper()}:\n"

                # Find max impact strength
                max_entropy_impact = 0
                max_unsure_impact = 0
                best_entropy_strength = None
                best_unsure_strength = None

                for strength, data in analysis[prompt_type][intervention_type].items():
                    abs_entropy = abs(data["mean_entropy_change"])
                    abs_unsure = abs(data["mean_unsure_change"])

                    if abs_entropy > max_entropy_impact:
                        max_entropy_impact = abs_entropy
                        best_entropy_strength = strength

                    if abs_unsure > max_unsure_impact:
                        max_unsure_impact = abs_unsure
                        best_unsure_strength = strength

                if best_entropy_strength:
                    data = analysis[prompt_type][intervention_type][
                        best_entropy_strength
                    ]
                    summary_text += f"  Max entropy impact: {data['mean_entropy_change']:.4f} ± {data['std_entropy_change']:.4f} (strength {best_entropy_strength})\n"

                if best_unsure_strength:
                    data = analysis[prompt_type][intervention_type][
                        best_unsure_strength
                    ]
                    summary_text += f"  Max unsure impact: {data['mean_unsure_change']:.4f} ± {data['std_unsure_change']:.4f} (strength {best_unsure_strength})\n"

        summary_text += "\n"

    ax_text.text(
        0.05,
        0.95,
        summary_text,
        transform=ax_text.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.suptitle(
        "Enhanced Causality Analysis: Adding vs Subtracting on Clean vs Conflict Prompts",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Enhanced causality analysis plot saved to {save_path}")
    else:
        plt.show()


def print_analysis_summary(analysis: Dict):
    """Print a summary of the analysis."""

    print("\n" + "=" * 80)
    print("ENHANCED CAUSALITY ANALYSIS SUMMARY")
    print("=" * 80)

    for prompt_type in ["clean", "conflict"]:
        print(f"\n{prompt_type.upper()} PROMPTS:")
        print("-" * 50)

        for intervention_type in ["add", "subtract"]:
            if intervention_type in analysis[prompt_type]:
                print(f"\n{intervention_type.upper()} INTERVENTION:")

                for strength in sorted(analysis[prompt_type][intervention_type].keys()):
                    data = analysis[prompt_type][intervention_type][strength]
                    print(f"  Strength {strength}:")
                    print(
                        f"    Entropy change: {data['mean_entropy_change']:.4f} ± {data['std_entropy_change']:.4f}"
                    )
                    print(
                        f"    Unsure change: {data['mean_unsure_change']:.4f} ± {data['std_unsure_change']:.4f}"
                    )
                    print(f"    Sample count: {data['count']}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced causality analysis")
    parser.add_argument(
        "--results",
        type=str,
        default="test_causality_results_32.pkl",
        help="Path to causality results pickle file",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="enhanced_causality_analysis.png",
        help="Path to save analysis plot",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results}...")
    results, _ = load_results(args.results)
    print(f"Loaded {len(results)} results")

    # Analyze prompt-intervention effects
    print("Analyzing prompt-intervention effects...")
    analysis = analyze_prompt_intervention_effects(results)

    # Print summary
    print_analysis_summary(analysis)

    # Create plot
    print(f"Creating enhanced plot...")
    plot_prompt_intervention_analysis(analysis, save_path=args.plot)


if __name__ == "__main__":
    main()
