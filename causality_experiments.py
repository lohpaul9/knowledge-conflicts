#!/usr/bin/env python3
"""
Causality Experiments for Knowledge Conflict Detection

This script implements causality experiments to test how manipulating
conflict direction vectors affects model behavior.
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
from dataclasses import dataclass
import scipy.stats as stats

import transformer_lens
from transformer_lens import HookedTransformer
from sklearn.preprocessing import StandardScaler


@dataclass
class CausalityResult:
    """Container for causality experiment results."""

    layer_name: str
    prompt_type: str  # "clean" or "conflict"
    experiment_type: str  # "entropy" or "unsure"
    intervention_type: str  # "add", "subtract", or "none"
    intervention_strength: float
    original_logits: np.ndarray
    modified_logits: np.ndarray
    original_probs: np.ndarray
    modified_probs: np.ndarray
    entropy_change: float
    unsure_prob_change: float
    answer_token_probs: Dict[str, float]


class CausalityExperimenter:
    """Run causality experiments by manipulating conflict direction vectors."""

    def __init__(self, model_name: str = "gemma-7b", device: str = "cuda"):
        """Initialize the causality experimenter."""
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"

        self.device = device
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        print(f"Model loaded successfully. Device: {device}")

        # Load trained classifiers
        self.classifiers = None
        self.conflict_directions = {}
        self.layer_names = []

    def load_classifiers(
        self, results_path: str, layer_range: Tuple[int, int] = (15, 25)
    ):
        """Load trained classifiers and extract conflict direction vectors."""
        print(f"Loading classifiers from {results_path}")
        with open(results_path, "rb") as f:
            results = pickle.load(f)

        self.classifiers = results["layer_classifiers"]

        # Filter to only the most relevant layers (15-25)
        start_layer, end_layer = layer_range
        self.layer_names = []
        for i in range(start_layer, end_layer + 1):
            layer_name = f"blocks.{i}.hook_resid_post"
            if layer_name in self.classifiers:
                self.layer_names.append(layer_name)

        print(f"Using layers {start_layer}-{end_layer}: {len(self.layer_names)} layers")

        # Extract and normalize conflict direction vectors
        print("Extracting conflict direction vectors...")
        for layer_name, classifier in self.classifiers.items():
            if layer_name not in self.layer_names:
                continue

            # Get the conflict direction vector (coefficients)
            conflict_direction = classifier.coef_[0]  # Shape: [feature_dim]

            # Normalize the vector
            norm = np.linalg.norm(conflict_direction)
            normalized_direction = conflict_direction / norm

            self.conflict_directions[layer_name] = {
                "original": conflict_direction,
                "normalized": normalized_direction,
                "norm": norm,
                "classifier": classifier,
            }

            print(f"  {layer_name}: norm = {norm:.3f}")

        print(f"Loaded {len(self.conflict_directions)} conflict direction vectors")

    def load_causal_dataset(
        self, dataset_path: str, max_examples: Optional[int] = None
    ):
        """Load the causal dataset."""
        print(f"Loading causal dataset from {dataset_path}")
        with open(dataset_path, "r") as f:
            data = json.load(f)

        examples = data["examples"]
        if max_examples and max_examples < len(examples):
            import random

            random.seed(42)
            examples = random.sample(examples, max_examples)

        print(f"Loaded {len(examples)} examples")
        return examples

    def format_prompt(
        self, prompt: str, question: str, options: List[str], add_unsure: bool = False
    ):
        """Format a prompt with question and options."""
        options_text = "\n".join(options)
        if add_unsure:
            options_text += "\nE. Unsure"
        formatted = f"{prompt}\n\n{question}\n{options_text}\n\nAnswer:"
        return formatted

    def get_answer_tokens(self, options: List[str], add_unsure: bool = False):
        """Get token IDs for answer options."""
        answer_tokens = {}

        for i, option in enumerate(options):
            option_text = f" {chr(65 + i)}"  # " A", " B", etc.
            tokens = self.model.to_tokens(option_text, prepend_bos=False)
            answer_tokens[chr(65 + i)] = tokens[0, 0].item()  # Get first token

        if add_unsure:
            unsure_text = " E"
            tokens = self.model.to_tokens(unsure_text, prepend_bos=False)
            answer_tokens["E"] = tokens[0, 0].item()

        return answer_tokens

    def compute_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of probability distribution."""
        # Add small epsilon to avoid log(0)
        probs = probs + 1e-10
        probs = probs / np.sum(probs)  # Renormalize
        return -np.sum(probs * np.log(probs))

    def intervention_hook(
        self, layer_name: str, intervention_type: str, strength: float
    ):
        """Create a hook that adds/subtracts the conflict direction vector."""

        def hook_fn(tensor, hook):
            # Get the conflict direction for this layer
            if layer_name not in self.conflict_directions:
                return tensor

            conflict_dir = self.conflict_directions[layer_name]["normalized"]

            # Apply intervention
            if intervention_type == "add":
                # Add conflict direction (increase conflict signal)
                intervention = strength * conflict_dir
            elif intervention_type == "subtract":
                # Subtract conflict direction (decrease conflict signal)
                intervention = -strength * conflict_dir
            else:
                return tensor

            # Apply intervention to the last token's activations
            # tensor shape: [batch, seq_len, feature_dim]
            modified_tensor = tensor.clone()
            modified_tensor[0, -1, :] += torch.tensor(
                intervention, device=tensor.device, dtype=tensor.dtype
            )

            return modified_tensor

        return hook_fn

    def run_single_experiment(
        self,
        example: Dict,
        layer_name: str,
        prompt_type: str,
        experiment_type: str,
        intervention_type: str,
        strength: float,
    ) -> CausalityResult:
        """Run a single causality experiment."""

        # Choose prompt
        if prompt_type == "clean":
            prompt = example["clean_prompt"]
        else:
            prompt = example["conflict_prompt"]

        # Format question
        add_unsure = experiment_type == "unsure"
        formatted_prompt = self.format_prompt(
            prompt, example["question"], example["options"], add_unsure
        )

        # Get answer tokens
        answer_tokens = self.get_answer_tokens(example["options"], add_unsure)

        # Run model without intervention
        with torch.no_grad():
            tokens = self.model.to_tokens(formatted_prompt)
            original_logits = self.model(tokens)
            original_probs = (
                torch.softmax(original_logits[0, -1, :], dim=-1).cpu().numpy()
            )

        # Run model with intervention
        intervention_hook = self.intervention_hook(
            layer_name, intervention_type, strength
        )

        with torch.no_grad():
            modified_logits = self.model.run_with_hooks(
                tokens, fwd_hooks=[(layer_name, intervention_hook)]
            )
            modified_probs = (
                torch.softmax(modified_logits[0, -1, :], dim=-1).cpu().numpy()
            )

        # Extract answer probabilities
        original_answer_probs = {
            opt: original_probs[token_id] for opt, token_id in answer_tokens.items()
        }
        modified_answer_probs = {
            opt: modified_probs[token_id] for opt, token_id in answer_tokens.items()
        }

        # Compute metrics
        original_entropy = self.compute_entropy(original_probs)
        modified_entropy = self.compute_entropy(modified_probs)
        entropy_change = modified_entropy - original_entropy

        unsure_prob_change = 0.0
        if add_unsure:
            original_unsure = original_answer_probs.get("E", 0.0)
            modified_unsure = modified_answer_probs.get("E", 0.0)
            unsure_prob_change = modified_unsure - original_unsure

        return CausalityResult(
            layer_name=layer_name,
            prompt_type=prompt_type,
            experiment_type=experiment_type,
            intervention_type=intervention_type,
            intervention_strength=strength,
            original_logits=original_logits[0, -1, :].cpu().numpy(),
            modified_logits=modified_logits[0, -1, :].cpu().numpy(),
            original_probs=original_probs,
            modified_probs=modified_probs,
            entropy_change=entropy_change,
            unsure_prob_change=unsure_prob_change,
            answer_token_probs=modified_answer_probs,
        )

    def run_causality_experiments(
        self,
        dataset_path: str,
        max_examples: int = 100,
        intervention_strengths: List[float] = None,
        batch_size: int = 4,
        fixed_layer: str = None,
    ):
        """Run comprehensive causality experiments with proper batching."""

        if intervention_strengths is None:
            intervention_strengths = [0.1, 0.5, 1.0, 2.0, 5.0]

        # Load dataset
        examples = self.load_causal_dataset(dataset_path, max_examples)

        # Initialize results storage
        all_results = []

        # Experiment parameters
        prompt_types = ["clean", "conflict"]
        experiment_types = ["entropy", "unsure"]
        intervention_types = ["add", "subtract"]

        # Use fixed layer if specified, otherwise use all layers
        layers_to_test = [fixed_layer] if fixed_layer else self.layer_names

        if fixed_layer and fixed_layer not in self.layer_names:
            print(
                f"Warning: Fixed layer '{fixed_layer}' not found in available layers. Available layers: {self.layer_names}"
            )
            return []

        total_experiments = (
            len(examples)
            * len(layers_to_test)
            * len(prompt_types)
            * len(experiment_types)
            * len(intervention_types)
            * len(intervention_strengths)
        )

        print(
            f"Running {total_experiments} experiments with batch size {batch_size}..."
        )
        if fixed_layer:
            print(f"Fixed layer: {fixed_layer}")
        else:
            print(f"Testing {len(layers_to_test)} layers")

        # Create all experiment configurations
        experiment_configs = []
        for layer_name in layers_to_test:
            for prompt_type in prompt_types:
                for experiment_type in experiment_types:
                    for intervention_type in intervention_types:
                        for strength in intervention_strengths:
                            experiment_configs.append(
                                {
                                    "layer_name": layer_name,
                                    "prompt_type": prompt_type,
                                    "experiment_type": experiment_type,
                                    "intervention_type": intervention_type,
                                    "strength": strength,
                                }
                            )

        # Process each config independently
        for config in tqdm(experiment_configs, desc="Processing experiment configs"):
            # Run this config with all examples batched
            config_results = self.run_single_config_experiment(config, examples)
            all_results.extend(config_results)

        return all_results

    def run_single_config_experiment(
        self, config: Dict, examples: List[Dict]
    ) -> List[CausalityResult]:
        """Run a single experiment configuration with all examples batched."""

        layer_name = config["layer_name"]
        prompt_type = config["prompt_type"]
        experiment_type = config["experiment_type"]
        intervention_type = config["intervention_type"]
        strength = config["strength"]

        # Create intervention hook
        intervention_hook = self.intervention_hook(
            layer_name, intervention_type, strength
        )

        # Prepare all prompts for this config across all examples
        prompts = []
        answer_tokens_list = []
        example_info = []

        for example in examples:
            # Choose prompt
            if prompt_type == "clean":
                prompt = example["clean_prompt"]
            else:
                prompt = example["conflict_prompt"]

            # Format question
            add_unsure = experiment_type == "unsure"
            formatted_prompt = self.format_prompt(
                prompt, example["question"], example["options"], add_unsure
            )

            # Get answer tokens
            answer_tokens = self.get_answer_tokens(example["options"], add_unsure)

            prompts.append(formatted_prompt)
            answer_tokens_list.append(answer_tokens)
            example_info.append(example)

        # Run model without intervention (batch)
        with torch.no_grad():
            # Tokenize all prompts at once
            all_tokens = self.model.to_tokens(prompts)
            original_logits = self.model(all_tokens)
            original_probs = (
                torch.softmax(original_logits[:, -1, :], dim=-1).cpu().numpy()
            )

        # Run model with intervention (batch)
        with torch.no_grad():
            modified_logits = self.model.run_with_hooks(
                all_tokens, fwd_hooks=[(layer_name, intervention_hook)]
            )
            modified_probs = (
                torch.softmax(modified_logits[:, -1, :], dim=-1).cpu().numpy()
            )

        # Process results for each example
        results = []
        for i, (example, answer_tokens) in enumerate(
            zip(example_info, answer_tokens_list)
        ):
            # Extract answer probabilities
            original_answer_probs = {
                opt: original_probs[i, token_id]
                for opt, token_id in answer_tokens.items()
            }
            modified_answer_probs = {
                opt: modified_probs[i, token_id]
                for opt, token_id in answer_tokens.items()
            }

            # Compute metrics
            original_entropy = self.compute_entropy(original_probs[i])
            modified_entropy = self.compute_entropy(modified_probs[i])
            entropy_change = modified_entropy - original_entropy

            unsure_prob_change = 0.0
            if experiment_type == "unsure":
                original_unsure = original_answer_probs.get("E", 0.0)
                modified_unsure = modified_answer_probs.get("E", 0.0)
                unsure_prob_change = modified_unsure - original_unsure

            result = CausalityResult(
                layer_name=layer_name,
                prompt_type=prompt_type,
                experiment_type=experiment_type,
                intervention_type=intervention_type,
                intervention_strength=strength,
                original_logits=original_logits[i, -1, :].cpu().numpy(),
                modified_logits=modified_logits[i, -1, :].cpu().numpy(),
                original_probs=original_probs[i],
                modified_probs=modified_probs[i],
                entropy_change=entropy_change,
                unsure_prob_change=unsure_prob_change,
                answer_token_probs=modified_answer_probs,
            )

            results.append(result)

        return results

    def analyze_results(self, results: List[CausalityResult]) -> Dict:
        """Analyze causality experiment results."""

        # Group results by experiment type
        entropy_results = [r for r in results if r.experiment_type == "entropy"]
        unsure_results = [r for r in results if r.experiment_type == "unsure"]

        analysis = {
            "entropy_analysis": self.analyze_entropy_results(entropy_results),
            "unsure_analysis": self.analyze_unsure_results(unsure_results),
            "summary_stats": {
                "total_experiments": len(results),
                "entropy_experiments": len(entropy_results),
                "unsure_experiments": len(unsure_results),
            },
        }

        return analysis

    def analyze_entropy_results(self, results: List[CausalityResult]) -> Dict:
        """Analyze entropy experiment results."""

        # Group by layer and intervention type
        layer_stats = {}

        for layer_name in self.layer_names:
            layer_results = [r for r in results if r.layer_name == layer_name]

            if not layer_results:
                continue

            # Group by intervention type
            add_results = [r for r in layer_results if r.intervention_type == "add"]
            subtract_results = [
                r for r in layer_results if r.intervention_type == "subtract"
            ]

            # Compute statistics
            layer_stats[layer_name] = {
                "add": {
                    "mean_entropy_change": np.mean(
                        [r.entropy_change for r in add_results]
                    ),
                    "std_entropy_change": np.std(
                        [r.entropy_change for r in add_results]
                    ),
                    "count": len(add_results),
                },
                "subtract": {
                    "mean_entropy_change": np.mean(
                        [r.entropy_change for r in subtract_results]
                    ),
                    "std_entropy_change": np.std(
                        [r.entropy_change for r in subtract_results]
                    ),
                    "count": len(subtract_results),
                },
            }

        return layer_stats

    def analyze_unsure_results(self, results: List[CausalityResult]) -> Dict:
        """Analyze unsure experiment results."""

        # Group by layer and intervention type
        layer_stats = {}

        for layer_name in self.layer_names:
            layer_results = [r for r in results if r.layer_name == layer_name]

            if not layer_results:
                continue

            # Group by intervention type
            add_results = [r for r in layer_results if r.intervention_type == "add"]
            subtract_results = [
                r for r in layer_results if r.intervention_type == "subtract"
            ]

            # Compute statistics
            layer_stats[layer_name] = {
                "add": {
                    "mean_unsure_change": np.mean(
                        [r.unsure_prob_change for r in add_results]
                    ),
                    "std_unsure_change": np.std(
                        [r.unsure_prob_change for r in add_results]
                    ),
                    "count": len(add_results),
                },
                "subtract": {
                    "mean_unsure_change": np.mean(
                        [r.unsure_prob_change for r in subtract_results]
                    ),
                    "std_unsure_change": np.std(
                        [r.unsure_prob_change for r in subtract_results]
                    ),
                    "count": len(subtract_results),
                },
            }

        return layer_stats

    def visualize_results(self, analysis: Dict, save_path: Optional[str] = None):
        """Visualize causality experiment results."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # Plot 1: Entropy changes by layer (add intervention)
        entropy_analysis = analysis["entropy_analysis"]
        layers = list(entropy_analysis.keys())
        add_entropy_changes = [
            entropy_analysis[layer]["add"]["mean_entropy_change"] for layer in layers
        ]

        ax1.bar(range(len(layers)), add_entropy_changes, color="red", alpha=0.7)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Mean Entropy Change")
        ax1.set_title("Entropy Change: Adding Conflict Direction")
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels([layer.split(".")[1] for layer in layers], rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="black", linestyle="--")

        # Plot 2: Entropy changes by layer (subtract intervention)
        subtract_entropy_changes = [
            entropy_analysis[layer]["subtract"]["mean_entropy_change"]
            for layer in layers
        ]

        ax2.bar(range(len(layers)), subtract_entropy_changes, color="blue", alpha=0.7)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Mean Entropy Change")
        ax2.set_title("Entropy Change: Subtracting Conflict Direction")
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels([layer.split(".")[1] for layer in layers], rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="black", linestyle="--")

        # Plot 3: Unsure probability changes by layer (add intervention)
        unsure_analysis = analysis["unsure_analysis"]
        add_unsure_changes = [
            unsure_analysis[layer]["add"]["mean_unsure_change"] for layer in layers
        ]

        ax3.bar(range(len(layers)), add_unsure_changes, color="orange", alpha=0.7)
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Mean Unsure Probability Change")
        ax3.set_title("Unsure Probability Change: Adding Conflict Direction")
        ax3.set_xticks(range(len(layers)))
        ax3.set_xticklabels([layer.split(".")[1] for layer in layers], rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color="black", linestyle="--")

        # Plot 4: Unsure probability changes by layer (subtract intervention)
        subtract_unsure_changes = [
            unsure_analysis[layer]["subtract"]["mean_unsure_change"] for layer in layers
        ]

        ax4.bar(range(len(layers)), subtract_unsure_changes, color="green", alpha=0.7)
        ax4.set_xlabel("Layer")
        ax4.set_ylabel("Mean Unsure Probability Change")
        ax4.set_title("Unsure Probability Change: Subtracting Conflict Direction")
        ax4.set_xticks(range(len(layers)))
        ax4.set_xticklabels([layer.split(".")[1] for layer in layers], rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color="black", linestyle="--")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Causality results plot saved to {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run causality experiments")
    parser.add_argument(
        "--classifier_results",
        type=str,
        required=True,
        help="Path to classifier results file",
    )
    parser.add_argument(
        "--causal_dataset",
        type=str,
        default="data/causal_1000.json",
        help="Path to causal dataset",
    )
    parser.add_argument(
        "--max_examples", type=int, default=50, help="Maximum examples to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="causality_results.pkl",
        help="Output file for results",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="causality_plot.png",
        help="Path to save visualization plot",
    )
    parser.add_argument(
        "--model", type=str, default="gemma-7b", help="Model name to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for experiments"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--fixed_layer",
        type=str,
        default=None,
        help="Fix to a specific layer (e.g., 'blocks.18.hook_resid_post')",
    )
    parser.add_argument(
        "--intervention_strengths",
        type=float,
        nargs="+",
        default=None,
        help="Custom intervention strengths (overrides default [0.1, 0.5, 1.0, 2.0, 5.0])",
    )

    args = parser.parse_args()

    # Initialize experimenter
    experimenter = CausalityExperimenter(model_name=args.model, device=args.device)

    # Load classifiers (only layers 15-25)
    experimenter.load_classifiers(args.classifier_results, layer_range=(15, 25))

    # Run experiments with batching
    results = experimenter.run_causality_experiments(
        args.causal_dataset,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        fixed_layer=args.fixed_layer,
        intervention_strengths=args.intervention_strengths,
    )

    # Analyze results
    analysis = experimenter.analyze_results(results)

    # Save results
    with open(args.output, "wb") as f:
        pickle.dump(
            {"results": results, "analysis": analysis, "experiment_config": vars(args)},
            f,
        )

    print(f"Results saved to {args.output}")

    # Create visualizations
    experimenter.visualize_results(analysis, save_path=args.plot)


if __name__ == "__main__":
    main()
