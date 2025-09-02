#!/usr/bin/env python3
"""
Layer Analysis for Knowledge Conflicts

This script analyzes which layers of the transformer are most sensitive
to detecting knowledge conflicts by testing classification accuracy across
different layers.
"""

import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import pickle

import transformer_lens
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA


class LayerAnalyzer:
    """Analyze which layers are most sensitive to conflicts."""

    def __init__(self, model_name: str = "gemma-7b", device: str = "cuda"):
        """Initialize the layer analyzer."""
        # Check if CUDA is available, fallback to CPU if not
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"

        self.device = device
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        print(f"Model loaded successfully. Device: {device}")

        # Get all available layers
        self.layer_names = self.get_layer_names()
        print(f"Found {len(self.layer_names)} layers to analyze")

    def get_layer_names(self) -> List[str]:
        """Get all available layer names for analysis."""
        # Get ALL available hook names from the model
        all_hooks = list(self.model.hook_dict.keys())

        # Filter to focus on the most relevant residual stream layers for Gemma 2 7B
        relevant_hooks = []

        for hook_name in all_hooks:
            # Include residual stream activations - these are the most informative for knowledge conflicts
            # if "hook_resid_pre" in hook_name:
            #     relevant_hooks.append(hook_name)
            # elif "hook_resid_mid" in hook_name:
            #     relevant_hooks.append(hook_name)
            if "hook_resid_post" in hook_name:
                relevant_hooks.append(hook_name)

        return relevant_hooks

    def load_dataset(
        self, dataset_path: str, max_examples: Optional[int] = None
    ) -> List[Dict]:
        """Load the conflict dataset from JSON file."""
        with open(dataset_path, "r") as f:
            data = json.load(f)
        examples = data["examples"]
        if max_examples and max_examples < len(examples):
            # Randomly sample examples instead of taking the first N
            import random

            random.seed(42)  # For reproducibility
            examples = random.sample(examples, max_examples)
        return examples

    def format_prompt(self, prompt: str, question: str, options: List[str]) -> str:
        """Format a prompt with question and options for the model."""
        options_text = "\n".join(options)
        formatted = f"{prompt}\n\n{question}\n{options_text}\n\nAnswer:"
        return formatted

    def extract_activations_for_layers(
        self, prompt: str, layer_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Extract activations from multiple layers for a given prompt."""
        activations = {}

        def save_activation(tensor, hook):
            activations[hook.name] = tensor.detach().clone()

        # Run the model with hooks
        with torch.no_grad():
            tokens = self.model.to_tokens(prompt)
            _ = self.model.run_with_hooks(
                tokens,
                fwd_hooks=[(layer_name, save_activation) for layer_name in layer_names],
            )

        return activations

    def analyze_layer_sensitivity(
        self, dataset_path: str, max_examples: int = 50
    ) -> Dict[str, float]:
        """Analyze which layers are most sensitive to conflicts."""
        print(f"Loading dataset from {dataset_path}")
        examples = self.load_dataset(dataset_path, max_examples)
        print(
            f"Analyzing {len(examples)} examples across {len(self.layer_names)} layers..."
        )

        # Store activation data for each layer
        layer_data = {
            layer: {"X": [], "y": [], "categories": []} for layer in self.layer_names
        }

        for example in tqdm(examples, desc="Processing examples"):
            try:
                # Format prompts
                clean_formatted = self.format_prompt(
                    example["clean_prompt"], example["question"], example["options"]
                )
                conflict_formatted = self.format_prompt(
                    example["conflict_prompt"], example["question"], example["options"]
                )

                # Extract activations for all layers
                clean_activations = self.extract_activations_for_layers(
                    clean_formatted, self.layer_names
                )
                conflict_activations = self.extract_activations_for_layers(
                    conflict_formatted, self.layer_names
                )

                # Process each layer
                for layer_name in self.layer_names:
                    if (
                        layer_name in clean_activations
                        and layer_name in conflict_activations
                    ):
                        # Get activations at the last token position
                        clean_act = (
                            clean_activations[layer_name][0, -1, :].cpu().numpy()
                        )
                        conflict_act = (
                            conflict_activations[layer_name][0, -1, :].cpu().numpy()
                        )

                        # For attention patterns, we need to handle the 2D shape differently
                        if "hook_pattern" in layer_name:
                            # For attention patterns, take the last row (attention to last token)
                            # and pad/truncate to a fixed size
                            clean_act = clean_act[-1, :]  # Shape: [seq_len]
                            conflict_act = conflict_act[-1, :]  # Shape: [seq_len]

                            # Pad or truncate to fixed size (e.g., 64)
                            max_len = 64
                            if clean_act.shape[0] > max_len:
                                clean_act = clean_act[:max_len]
                                conflict_act = conflict_act[:max_len]
                            else:
                                # Pad with zeros
                                pad_len = max_len - clean_act.shape[0]
                                clean_act = np.pad(clean_act, (0, pad_len), "constant")
                                conflict_act = np.pad(
                                    conflict_act, (0, pad_len), "constant"
                                )

                        # Flatten activations to ensure consistent shape
                        clean_act_flat = clean_act.flatten()
                        conflict_act_flat = conflict_act.flatten()

                        # Ensure both activations have the same shape
                        if clean_act_flat.shape == conflict_act_flat.shape:
                            # Add both clean and conflict examples
                            layer_data[layer_name]["X"].append(
                                clean_act_flat
                            )  # 0 for clean
                            layer_data[layer_name]["y"].append(0)
                            layer_data[layer_name]["categories"].append(
                                example["category"]
                            )

                            layer_data[layer_name]["X"].append(
                                conflict_act_flat
                            )  # 1 for conflict
                            layer_data[layer_name]["y"].append(1)
                            layer_data[layer_name]["categories"].append(
                                example["category"]
                            )

            except Exception as e:
                print(f"Error processing example: {e}")
                continue

        # Train classifiers for each layer
        layer_accuracies = {}
        layer_classifiers = {}
        layer_data_info = {}

        for layer_name in tqdm(self.layer_names, desc="Training classifiers"):
            data = layer_data[layer_name]

            if len(data["X"]) < 20:  # Need enough examples
                print(
                    f"Skipping {layer_name}: insufficient examples ({len(data['X'])})"
                )
                continue

            # Convert to numpy arrays and check shapes
            try:
                X = np.array(data["X"])
                y = np.array(data["y"])
                categories = data["categories"]

                print(f"Layer {layer_name}: X shape {X.shape}, y shape {y.shape}")

                # Check if we have both classes
                unique_classes = np.unique(y)
                if len(unique_classes) < 2:
                    print(
                        f"Skipping {layer_name}: only one class present {unique_classes}"
                    )
                    continue

                # Check class balance
                class_counts = np.bincount(y)
                print(f"Layer {layer_name}: class counts {class_counts}")

                # Train classifier
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )

                classifier = LogisticRegression(random_state=42, max_iter=1000)
                classifier.fit(X_train, y_train)

                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Store results
                layer_accuracies[layer_name] = accuracy
                layer_classifiers[layer_name] = classifier
                layer_data_info[layer_name] = {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "categories": categories,
                    "class_counts": class_counts,
                    "feature_dim": X.shape[1],
                }

                print(f"Layer {layer_name}: accuracy = {accuracy:.3f}")

            except Exception as e:
                print(f"Error processing {layer_name}: {e}")
                continue

        return layer_accuracies, layer_classifiers, layer_data_info

    def visualize_layer_sensitivity(
        self, layer_accuracies: Dict[str, float], save_path: Optional[str] = None
    ):
        """Visualize layer sensitivity to conflicts."""
        if not layer_accuracies:
            print("No layer accuracies to visualize")
            return

        # Sort layers by accuracy
        sorted_layers = sorted(
            layer_accuracies.items(), key=lambda x: x[1], reverse=True
        )

        # Create visualization - make it wider for better readability
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Bar chart of accuracies
        layer_names = [layer for layer, _ in sorted_layers]
        accuracies = [acc for _, acc in sorted_layers]

        # Create better labels for layer names - show block number and hook type
        def format_layer_name(name):
            if "blocks." in name:
                parts = name.split(".")
                if len(parts) >= 3:
                    block_num = parts[1]
                    hook_type = parts[-1].replace("hook_", "")
                    return f"B{block_num}.{hook_type}"
                else:
                    return name.replace("hook_", "")
            else:
                return name.replace("hook_", "")

        bars = ax1.bar(range(len(layer_names)), accuracies, color="skyblue", alpha=0.7)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Classification Accuracy")
        ax1.set_title("Layer Sensitivity to Knowledge Conflicts")
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(
            [format_layer_name(name) for name in layer_names],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random (0.5)")
        ax1.legend()

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Plot 2: Top 10 layers
        top_10 = sorted_layers[:10]
        top_names = [layer for layer, _ in top_10]
        top_accs = [acc for _, acc in top_10]

        bars2 = ax2.bar(range(len(top_names)), top_accs, color="lightcoral", alpha=0.7)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Classification Accuracy")
        ax2.set_title("Top 10 Most Sensitive Layers")
        ax2.set_xticks(range(len(top_names)))
        ax2.set_xticklabels(
            [format_layer_name(name) for name in top_names],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random (0.5)")
        ax2.legend()

        # Add value labels on bars
        for bar, acc in zip(bars2, top_accs):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Layer sensitivity plot saved to {save_path}")
        else:
            plt.show()

        # Print top layers
        print("\nTop 10 most sensitive layers:")
        for i, (layer, acc) in enumerate(top_10):
            print(f"  {i+1:2d}. {layer}: {acc:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze layer sensitivity to knowledge conflicts"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--model", type=str, default="gemma-7b", help="Model name to use"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--max_examples", type=int, default=50, help="Maximum examples to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="layer_analysis_results.pkl",
        help="Output file for results",
    )
    parser.add_argument(
        "--plot", type=str, default=None, help="Path to save visualization plot"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = LayerAnalyzer(model_name=args.model, device=args.device)

    # Analyze layer sensitivity
    layer_accuracies, layer_classifiers, layer_data_info = (
        analyzer.analyze_layer_sensitivity(args.dataset, args.max_examples)
    )

    # Save results
    results = {
        "layer_accuracies": layer_accuracies,
        "layer_classifiers": layer_classifiers,
        "layer_data_info": layer_data_info,
        "model_name": analyzer.model.cfg.model_name,
        "device": analyzer.device,
        "dataset_path": args.dataset,
        "max_examples": args.max_examples,
        "layer_names": analyzer.layer_names,
    }

    with open(args.output, "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {args.output}")

    # Create visualizations
    if args.plot:
        analyzer.visualize_layer_sensitivity(layer_accuracies, save_path=args.plot)
    else:
        analyzer.visualize_layer_sensitivity(layer_accuracies)


if __name__ == "__main__":
    main()
