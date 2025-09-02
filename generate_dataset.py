#!/usr/bin/env python3
"""
Dataset Generator for Intra-Context Knowledge Conflicts

This script uses OpenAI's GPT-5 mini to generate synthetic datasets for detecting
knowledge conflicts in LLMs using mechanistic interpretability approaches.
"""

import json
import os
import random
import asyncio
import aiohttp
from typing import List, Dict, Any, Tuple
from openai import OpenAI, AsyncOpenAI
from dataclasses import dataclass
from datetime import datetime
import time
from asyncio import Semaphore
from collections import deque
from pydantic import BaseModel, Field

# Set up OpenAI client (will be initialized when needed)
client = None
async_client = None

# Default rate limiting: 200 requests per minute = 1 request every 0.3 seconds
DEFAULT_RATE_LIMIT_PER_MINUTE = 200
RATE_LIMIT_INTERVAL = 60.0 / DEFAULT_RATE_LIMIT_PER_MINUTE


# Pydantic models for structured outputs
# These models define the schema that will be passed to OpenAI for structured outputs
class ConflictExampleItem(BaseModel):
    """Individual conflict example item."""

    clean_statement: str = Field(
        description="A standalone factual statement without contradictions. It should be a statement that makes sense on its own, but has enough space so that it could be altered to contain a contradiction but have the same structure so we can alter it to contain a contradiction."
    )
    conflict_statement: str = Field(
        description="A standalone statement very similar to the clean statement, but altered to contain a contradiction. The clean and conflict statements should be almost identical except for the contradiction."
    )
    question: str = Field(
        description="Multiple choice question that tests a fact from the statement. This should be designed to work with EITHER the clean statement OR the conflict statement independently. This question should have a clearly correct answer for the clean statement, but should have no clear answer for the conflict statement."
    )
    options: List[str] = Field(
        description="Four answer options formatted as 'A. [text]', 'B. [text]', etc."
    )
    correct_answer_for_clean_statement: str = Field(
        description="The correct answer given that we use the clean statement as a single letter: A, B, C, or D"
    )


class ConflictExamplesResponse(BaseModel):
    """Response structure for conflict examples."""

    examples: List[ConflictExampleItem] = Field(
        description="List of generated conflict examples",
        # min_length=1,
        # max_length=20,  # Reasonable upper limit
    )


class RateLimiter:
    """Rate limiter to ensure we don't exceed API limits."""

    def __init__(self, max_requests_per_minute: int):
        self.max_requests_per_minute = max_requests_per_minute
        self.interval = 60.0 / max_requests_per_minute
        self.request_times = deque()
        self.semaphore = Semaphore(max_requests_per_minute)

    async def acquire(self):
        """Acquire permission to make a request."""
        await self.semaphore.acquire()

        now = time.time()

        # Remove old requests outside the 1-minute window
        while self.request_times and now - self.request_times[0] >= 60.0:
            self.request_times.popleft()

        # If we've made too many requests recently, wait
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60.0 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.request_times.append(time.time())

    def release(self):
        """Release the semaphore after request completion."""
        self.semaphore.release()


# Global rate limiter (will be initialized with CLI argument)
rate_limiter = None


@dataclass
class ConflictExample:
    """Represents a single conflict example with clean and conflict prompts."""

    clean_prompt: str
    conflict_prompt: str
    question: str
    options: List[str]
    correct_answer_for_clean_statement: str
    category: str
    conflict_type: str


class DatasetGenerator:
    def __init__(self):
        self.categories = {
            "factual_contradictions": {
                "description": "Factual contradictions where the prompt asserts conflicting facts about the same subject. These should be based on common place knowledge that most people would know (e.g., famous landmarks, basic geography, well-known historical facts, common scientific facts). Avoid obscure or specialized knowledge.",
                "examples": [
                    {
                        "clean_statement": "The ",
                        "conflict_statement": "The Eiffel Tower is in Berlin.",
                        "question": "Where is the Eiffel Tower located?",
                        "options": ["Paris", "Berlin", "London", "Rome"],
                        "correct_answer_for_clean_statement": "Paris",
                    },
                    {
                        "clean_statement": "Mount Everest is 8,848 meters tall.",
                        "conflict_statement": "Mount Everest is 9,000 meters tall.",
                        "question": "What is the height of Mount Everest?",
                        "options": [
                            "9,000 meters",
                            "8,000 meters",
                            "8,848 meters",
                            "10,000 meters",
                        ],
                        "correct_answer_for_clean_statement": "8,848 meters",
                    },
                ],
            },
            "temporal_contradictions": {
                "description": "Conflicts in time or sequence where events are impossible in the given order",
                "examples": [
                    {
                        "clean_statement": "She was born in 1990 and graduated in 2012.",
                        "conflict_statement": "She was born in 1990 and graduated in 1980.",
                        "question": "In what year did she graduate?",
                        "options": ["1980", "1995", "2012", "2005"],
                        "correct_answer_for_clean_statement": "2012",
                    },
                    {
                        "clean_statement": "The movie was released in 2020 but filmed in 2019.",
                        "conflict_statement": "The movie was released in 2020 but filmed in 2025.",
                        "question": "In what year was the movie filmed?",
                        "options": ["2019", "2025", "2018", "2021"],
                        "correct_answer_for_clean_statement": "2019",
                    },
                ],
            },
            "negation_contradictions": {
                "description": "Negated statements that contradict positive ones about the same subject",
                "examples": [
                    {
                        "clean_statement": "Alice is a pilot. She flies planes daily.",
                        "conflict_statement": "Alice is not a pilot. She flies planes daily.",
                        "question": "What is Alice's profession?",
                        "options": [
                            "Engineer",
                            "Flight attendant",
                            "Air traffic controller",
                            "Pilot",
                        ],
                        "correct_answer_for_clean_statement": "Pilot",
                    },
                    {
                        "clean_statement": "The restaurant is open on Sundays. It serves brunch every Sunday.",
                        "conflict_statement": "The restaurant is not open on Sundays. It serves brunch every Sunday.",
                        "question": "Is the restaurant open on Sundays?",
                        "options": [
                            "No",
                            "Yes",
                            "Only for brunch",
                            "Depends on the week",
                        ],
                        "correct_answer_for_clean_statement": "Yes",
                    },
                ],
            },
            "role_attribute_contradictions": {
                "description": "Conflicting attributes or categories assigned to the same entity. Note that each statement should be contradictory in a self-contained way. For example, Clean: 'Bob is a doctor' and Conflict: 'Bob is a lawyer' are not contradictory because they are not self-contained.",
                "examples": [
                    {
                        "clean_statement": "Bob, a doctor, told me about his job as a doctor.",
                        "conflict_statement": "Bob, a doctor, told me about his job as a lawyer.",
                        "question": "What is Bob's profession?",
                        "options": [
                            "Lawyer",
                            "Doctor",
                            "Engineer",
                            "Neither",
                        ],
                        "correct_answer_for_clean_statement": "Doctor",
                    },
                    {
                        "clean_statement": "Maria, who is 25 years old, celebrated her 25th birthday.",
                        "conflict_statement": "Maria, who is 25 years old, celebrated her 50th birthday.",
                        "question": "How old is Maria?",
                        "options": [
                            "50 years old",
                            "30 years old",
                            "25 years old",
                            "20 years old",
                        ],
                        "correct_answer_for_clean_statement": "25 years old",
                    },
                ],
            },
        }

    async def generate_conflict_examples_async(
        self,
        category: str,
        num_examples: int,
        batch_idx: int,
        num_batches: int,
        rate_limit_per_minute: int,
    ) -> List[Dict[str, Any]]:
        """Generate conflict examples for a specific category using GPT-5 mini (async version)."""

        # Initialize async client if not already done
        global async_client, rate_limiter
        if async_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            async_client = AsyncOpenAI(api_key=api_key)

        # Initialize rate limiter if not already done
        if rate_limiter is None:
            rate_limiter = RateLimiter(rate_limit_per_minute)

        category_info = self.categories[category]

        # Format examples as JSON for better clarity
        example_json = json.dumps(category_info["examples"], indent=2)

        system_prompt = f"""You are an expert at creating intra-context knowledge conflicts for research on mechanistic interpretability.

Your task is to generate {num_examples} pairs of statements where:
1. The clean statement is factual and consistent
2. The conflict statement is almost identical but contains a contradiction
3. The statements should be natural and realistic
4. The contradiction should be clear and testable

IMPORTANT: The multiple choice question should be designed to work with EITHER the clean statement OR the conflict statement independently. The question should test a fact about the preceding statement, not reference both statements or ask about contradictions.
The statements should be widely varied in topic and content. Come up with new topics and content for each example.

IMPORTANT: The conflict statement should be contradictory in a self-contained way. For example, Clean: 'Bob is a doctor' and Conflict: 'Bob is a lawyer' are not contradictory because they are not self-contained. 
However, Clean: 'Bob is a doctor who does heart surgery' and Conflict: 'Bob is a lawyer who does heart surgery' are contradictory because they are self-contained.

IMPORTANT: Questions should only contain a correct answer for the clean statement. There should not be a clearly correct answer for the conflict statement, because the statement contains a contradiction.

You should NOT be using any of the examples provided in the examples of this category.

Category: {category}
Description: {category_info['description']}

Examples of this category (exactly as you should format your output):
{example_json}

For each example, you must also create:
- A multiple choice question that tests a fact from the statement
- 4 answer options (A, B, C, D) where one is clearly correct
- The correct answer letter

The clean and conflict statements should be almost identical except for the contradiction.

Please respond in JSON format with the following structure:
{{
  "examples": [
    {{
      "clean_statement": "clean statement here",
      "conflict_statement": "conflict statement here", 
      "question": "multiple choice question here",
      "options": ["option A", "option B", "option C", "option D"],
      "correct_answer_for_clean_statement": "C"
    }}
  ]
}}"""

        user_prompt = f"""Generate {num_examples} examples for the {category} category.

Each example should include:
1. A clean statement (factual, no contradictions)
2. A conflict statement (almost identical but with a contradiction)
3. A multiple choice question that tests a fact from the statement
4. Four answer options formatted as "A. [text]", "B. [text]", "C. [text]", "D. [text]"
5. The correct answer as a single letter: "A", "B", "C", or "D"


CRITICAL: 
- The question should work with either statement independently. It should test a fact about the statement, not ask about contradictions between statements.
- Options must be formatted with letter prefixes: "A. [text]", "B. [text]", etc.
- Correct answer must be a single letter: "A", "B", "C", or "D"
- The statements should be widely varied in topic and content. Come up with new topics and content for each example.

Make sure the statements are natural, realistic, and the contradiction is clear.

Please provide your response in JSON format as specified."""

        try:
            # Acquire rate limit permission
            await rate_limiter.acquire()

            response = await async_client.chat.completions.parse(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=ConflictExamplesResponse,
                max_completion_tokens=128000,
            )

            result = response.choices[0].message.parsed
            # Convert to the expected format
            examples = []
            for example in result.examples:
                examples.append(
                    {
                        "clean_prompt": example.clean_statement,
                        "conflict_prompt": example.conflict_statement,
                        "question": example.question,
                        "options": example.options,
                        "correct_answer_for_clean_statement": example.correct_answer_for_clean_statement,
                        "category": category,
                        "conflict_type": self._get_conflict_type(category),
                    }
                )

            print(
                f"Completed {category} batch {batch_idx}/{num_batches} with {len(examples)} examples"
            )

            return examples

        except Exception as e:
            print(f"Error generating examples for {category}: {e}")
            return []
        finally:
            rate_limiter.release()

    def generate_conflict_examples(
        self, category: str, num_examples: int
    ) -> List[Dict[str, Any]]:
        """Generate conflict examples for a specific category using GPT-5 mini (sync version)."""

        # Initialize client if not already done
        global client
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            client = OpenAI(api_key=api_key)

        category_info = self.categories[category]

        # Format examples as JSON for better clarity
        example_json = json.dumps(category_info["examples"], indent=2)

        system_prompt = f"""You are an expert at creating intra-context knowledge conflicts for research on mechanistic interpretability.

Your task is to generate {num_examples} pairs of statements where:
1. The clean statement is factual and consistent
2. The conflict statement is almost identical but contains a contradiction
3. The statements should be natural and realistic
4. The contradiction should be clear and testable

IMPORTANT: The multiple choice question should be designed to work with EITHER the clean statement OR the conflict statement independently. The question should test a fact about the preceding statement, not reference both statements or ask about contradictions.
The statements should be widely varied in topic and content. Come up with new topics and content for each example.

Category: {category}
Description: {category_info['description']}

Examples of this category (exactly as you should format your output):
{example_json}

For each example, you must also create:
- A multiple choice question that tests a fact from the statement
- 4 answer options (A, B, C, D) where one is clearly correct
- The correct answer letter

The clean and conflict statements should be almost identical except for the contradiction.

You should NOT be using any of the examples provided in the examples of this category.

Please respond in JSON format with the following structure:
{{
  "examples": [
    {{
      "clean_statement": "clean statement here",
      "conflict_statement": "conflict statement here", 
      "question": "multiple choice question here",
      "options": ["option A", "option B", "option C", "option D"],
      "correct_answer_for_clean_statement": "C"
    }}
  ]
}}"""

        user_prompt = f"""Generate {num_examples} examples for the {category} category.

Each example should include:
1. A clean statement (factual, no contradictions)
2. A conflict statement (almost identical but with a contradiction)
3. A multiple choice question that tests a fact from the statement
4. Four answer options formatted as "A. [text]", "B. [text]", "C. [text]", "D. [text]"
5. The correct answer as a single letter: "A", "B", "C", or "D"

CRITICAL: 
- The question should work with either statement independently. It should test a fact about the statement, not ask about contradictions between statements.
- Options must be formatted with letter prefixes: "A. [text]", "B. [text]", etc.
- Correct answer must be a single letter: "A", "B", "C", or "D"
- The statements should be widely varied in topic and content. Come up with new topics and content for each example.

Make sure the statements are natural, realistic, and the contradiction is clear.

Please provide your response in JSON format as specified."""

        try:
            response = client.chat.completions.parse(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=128000,
                response_format=ConflictExamplesResponse,
            )

            # Handle structured output response
            try:
                result = response.choices[0].message.parsed

                # Validate that we got the expected number of examples
                if len(result.examples) != num_examples:
                    print(
                        f"Warning: Expected {num_examples} examples, but got {len(result.examples)} for {category}"
                    )
                    # If we got fewer examples than expected, we could retry or accept what we have
                    # For now, we'll accept what we got but log the issue

                # Convert to the expected format
                examples = []
                for example in result.examples:
                    examples.append(
                        {
                            "clean_prompt": example.clean_statement,
                            "conflict_prompt": example.conflict_statement,
                            "question": example.question,
                            "options": example.options,
                            "correct_answer_for_clean_statement": example.correct_answer_for_clean_statement,
                            "category": category,
                            "conflict_type": self._get_conflict_type(category),
                        }
                    )

                return examples

            except Exception as validation_error:
                print(f"Validation error for {category}: {validation_error}")

                return examples

        except Exception as e:
            print(f"Error generating examples for {category}: {e}")
            return []

    def _get_conflict_type(self, category: str) -> str:
        """Get a more specific conflict type description."""
        type_mapping = {
            "factual_contradictions": "factual_inconsistency",
            "temporal_contradictions": "temporal_inconsistency",
            "negation_contradictions": "logical_negation_conflict",
            "role_attribute_contradictions": "attribute_assignment_conflict",
        }
        return type_mapping.get(category, "unknown_conflict")

    async def generate_dataset_async(
        self,
        max_examples: int,
        output_file: str = "knowledge_conflicts_dataset.json",
        batch_size: int = 16,
        rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
    ) -> List[ConflictExample]:
        """Generate the complete dataset using GPT-5 mini with parallel processing."""
        print(
            f"Generating dataset with {max_examples} examples using parallel processing..."
        )
        print(f"Rate limit: {rate_limit_per_minute} requests per minute")
        print(f"Batch size: {batch_size} examples per request")
        print(f"Note: Rate limiting only applies to parallel processing mode")

        # Calculate examples per category
        examples_per_category = max_examples // len(self.categories)
        remaining = max_examples % len(self.categories)

        # Create batch tasks for all categories
        # This approach scales better: instead of one large task per category,
        # we create multiple smaller batch tasks that can run in parallel
        tasks = []
        task_info = []  # Track task metadata for better logging

        for category_name in self.categories.keys():
            category_examples = examples_per_category + (1 if remaining > 0 else 0)
            remaining = max(0, remaining - 1)

            # Split category examples into batches
            num_batches = (
                category_examples + batch_size - 1
            ) // batch_size  # Ceiling division

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, category_examples)
                batch_size_actual = end_idx - start_idx

                print(
                    f"Queuing batch {batch_idx + 1}/{num_batches} for {category_name}: {batch_size_actual} examples"
                )

                # Create async task for this batch
                task = self.generate_conflict_examples_async(
                    category_name,
                    batch_size_actual,
                    batch_idx,
                    num_batches,
                    rate_limit_per_minute,
                )
                tasks.append(task)
                task_info.append(
                    {
                        "category": category_name,
                        "batch_idx": batch_idx + 1,
                        "num_batches": num_batches,
                        "batch_size": batch_size_actual,
                    }
                )

        # Execute all tasks concurrently
        print(f"Starting parallel generation with {len(tasks)} total batches...")
        all_examples = []

        # Use asyncio.gather to run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            task_meta = task_info[i]
            category_name = task_meta["category"]
            batch_info = f"batch {task_meta['batch_idx']}/{task_meta['num_batches']}"

            if isinstance(result, Exception):
                print(f"❌ Error generating {batch_info} for {category_name}: {result}")
            elif result:
                all_examples.extend(result)
                print(
                    f"✅ Generated {len(result)} examples for {category_name} ({batch_info})"
                )
            else:
                print(f"❌ Failed to generate {batch_info} for {category_name}")

        # Convert to ConflictExample objects
        dataset = []
        for example_data in all_examples[:max_examples]:
            example = ConflictExample(
                clean_prompt=example_data["clean_prompt"],
                conflict_prompt=example_data["conflict_prompt"],
                question=example_data["question"],
                options=example_data["options"],
                correct_answer_for_clean_statement=example_data[
                    "correct_answer_for_clean_statement"
                ],
                category=example_data["category"],
                conflict_type=example_data["conflict_type"],
            )
            dataset.append(example)

        # Save to JSON
        self._save_to_json(dataset, output_file)

        print(f"Dataset generated successfully! Saved to {output_file}")
        print(
            f"Generated {len(dataset)} examples across {len(set(ex.category for ex in dataset))} categories"
        )

        return dataset

    def generate_dataset(
        self,
        max_examples: int,
        output_file: str = "knowledge_conflicts_dataset.json",
    ) -> List[ConflictExample]:
        """Generate the complete dataset using GPT-5 mini (sync version)."""
        print(f"Generating dataset with {max_examples} examples...")

        # Calculate examples per category
        examples_per_category = max_examples // len(self.categories)
        remaining = max_examples % len(self.categories)

        all_examples = []

        for category_name in self.categories.keys():
            category_examples = examples_per_category + (1 if remaining > 0 else 0)
            remaining = max(0, remaining - 1)

            print(f"Generating {category_examples} examples for {category_name}...")

            # Generate examples for this category
            examples = self.generate_conflict_examples(category_name, category_examples)

            if examples:
                all_examples.extend(examples)
                print(f"✅ Generated {len(examples)} examples for {category_name}")
            else:
                print(f"❌ Failed to generate examples for {category_name}")

            # Add a small delay to avoid rate limiting
            time.sleep(1)

        # Convert to ConflictExample objects
        dataset = []
        for example_data in all_examples[:max_examples]:
            example = ConflictExample(
                clean_prompt=example_data["clean_prompt"],
                conflict_prompt=example_data["conflict_prompt"],
                question=example_data["question"],
                options=example_data["options"],
                correct_answer_for_clean_statement=example_data[
                    "correct_answer_for_clean_statement"
                ],
                category=example_data["category"],
                conflict_type=example_data["conflict_type"],
            )
            dataset.append(example)

        # Save to JSON
        self._save_to_json(dataset, output_file)

        print(f"Dataset generated successfully! Saved to {output_file}")
        print(
            f"Generated {len(dataset)} examples across {len(set(ex.category for ex in dataset))} categories"
        )

        return dataset

    def _save_to_json(self, dataset: List[ConflictExample], output_file: str):
        """Save the dataset to a JSON file."""
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_examples": len(dataset),
                "categories": list(set(ex.category for ex in dataset)),
                "description": "Intra-context knowledge conflict dataset for mechanistic interpretability research",
            },
            "examples": [],
        }

        for example in dataset:
            data["examples"].append(
                {
                    "clean_prompt": example.clean_prompt,
                    "conflict_prompt": example.conflict_prompt,
                    "question": example.question,
                    "options": example.options,
                    "correct_answer_for_clean_statement": example.correct_answer_for_clean_statement,
                    "category": example.category,
                    "conflict_type": example.conflict_type,
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


async def main_async(batch_size: int = 16):
    """Async main function to run the dataset generation with parallel processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate knowledge conflict dataset using GPT-5-mini with parallel processing"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20,
        help="Maximum number of examples to generate (default: 20)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="knowledge_conflicts_dataset.json",
        help="Output JSON file name (default: knowledge_conflicts_dataset.json)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing (default: True for async main)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of examples to generate per API request (default: 16)",
    )
    parser.add_argument(
        "--rate_limit",
        type=int,
        default=DEFAULT_RATE_LIMIT_PER_MINUTE,
        help=f"Rate limit requests per minute (default: {DEFAULT_RATE_LIMIT_PER_MINUTE})",
    )

    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return

    # Generate the dataset
    generator = DatasetGenerator()
    dataset = await generator.generate_dataset_async(
        args.max_examples, args.output_file, batch_size, args.rate_limit
    )

    # Print summary
    print("\n" + "=" * 50)
    print("DATASET GENERATION COMPLETE (Parallel Mode)")
    print("=" * 50)
    print(f"Total examples: {len(dataset)}")

    # Count by category
    category_counts = {}
    for example in dataset:
        category_counts[example.category] = category_counts.get(example.category, 0) + 1

    print("\nExamples by category:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

    print(f"\nDataset saved to: {args.output_file}")


def main():
    """Main function to run the dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate knowledge conflict dataset using GPT-5-mini"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20,
        help="Maximum number of examples to generate (default: 20)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="knowledge_conflicts_dataset.json",
        help="Output JSON file name (default: knowledge_conflicts_dataset.json)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing (default: False)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of examples to generate per API request (only used in parallel mode, default: 16)",
    )
    parser.add_argument(
        "--rate_limit",
        type=int,
        default=DEFAULT_RATE_LIMIT_PER_MINUTE,
        help=f"Rate limit requests per minute (only used in parallel mode, default: {DEFAULT_RATE_LIMIT_PER_MINUTE})",
    )

    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return

    if args.parallel:
        # Run async version with batch size
        asyncio.run(main_async(args.batch_size))
    else:
        # Generate the dataset (sync version)
        generator = DatasetGenerator()
        dataset = generator.generate_dataset(args.max_examples, args.output_file)

        # Print summary
        print("\n" + "=" * 50)
        print("DATASET GENERATION COMPLETE")
        print("=" * 50)
        print(f"Total examples: {len(dataset)}")

        # Count by category
        category_counts = {}
        for example in dataset:
            category_counts[example.category] = (
                category_counts.get(example.category, 0) + 1
            )

        print("\nExamples by category:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")

        print(f"\nDataset saved to: {args.output_file}")


if __name__ == "__main__":
    main()
