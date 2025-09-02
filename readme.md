# Detecting Knowledge Conflicts in Language Models Using Linear Probes

This research investigates whether contradictions in model prompts can be detected through simple linear probes on transformer activations. Detecting knowledge conflicts is crucial for multiple reasons: contradictory information in model contexts can lead to hallucinations, degraded reasoning, and potential safety vulnerabilities when users attempt to override system prompts. While prior work has explored how to steer models between conflicting knowledge sources (e.g., JuiCE) or characterized knowledge conflicts theoretically, robust detection methods remain understudied.

# Run instructions

1. Install dependencies: `pip install -r requirements.txt`

2. Set up OpenAI API key: `export OPENAI_API_KEY=<your_api_key>`

3. Run dataset generation: `generate_dataset.py`

4. Run layer analysis to generate an analysis for every layer in the model: `layer_analysis.py`

5. Get analysis for a single layer: `analyse_single_layer.py`

6. Run causality experiments to see if chosen direction vectors for conflict are causal using proxies: `causality_experiments.py`

# Research Description

Research goal: See if we can detect knowledge conflicts for LLMs that happen within the prompt, using mechanistic interpretability approaches. As a first approach, we will focus on intra-context knowledge conflicts.

We first focus on answering the question on _can we detect conflict_ before we move on to making further extensions. This involves a few key steps.

## Part 1: Dataset Creation

Create a intra-context knowledge conflict synthetic dataset. This should probably be pairs of prompts, whereby the prompts are almost exactly the same except for one phrase which contains a conflict. The prompt should also end with making the model output some multiple choice, I would think. For example:
Prompt conflict: "Bob, a doctor, told me about his job as a lawyer."
Prompt clean: "Bob, a doctor, told me about his job as a doctor."

Hard requirement: The prompts should be ALMOST THE SAME except for the conflict. I want to control for everything else, so even the prompt structure should be the same.

I've also identified some categories of conflicts that I want to encode in the dataset. These are:

1. Factual Contradictions
   - The prompt asserts a fact that conflicts with another fact in the same context (e.g., “The Eiffel Tower is in Berlin… Paris”).
   - These test the model’s stored world knowledge and factual consistency.
2. Temporal Contradictions
   - Conflicts in time or sequence (e.g., “She was born in 1990 and graduated in 1980”).
   - These test chronological reasoning and temporal consistency.
3. Negation-Based Contradictions
   - A negated statement contradicts a positive one (e.g., “Alice is not a pilot. She flies planes daily.”).
   - These test the model’s ability to handle negation logically.
4. Role/Attribute Contradictions
   - Conflicting attributes or categories are assigned to the same entity (e.g., “Bob is a doctor… Bob’s job as a lawyer”).
   - More general than just “roles”: it includes jobs, nationalities, physical attributes, etc

The prompts should also come with a question after that is a multiple choice question. This can be generated separately from the list of question. It should be a multiple choice question that is directly related to the conflict. E.g.:

"Bob, a doctor, told me about his job as a doctor."
Question: What does bob work as?
Answer: (a) Lawyer (b) Doctor (c) Engineer

## Part 2: Mechanistic Interpretability

For this part, we want to see if we can detect conflict using mechanistic interpretability approaches.

### Detecting conflict

Run this through a transformer model, and check to see at its layers and activations (of I think, the last token), whether I'm able to strongly detect a vector that is able to represent 'conflict'. We should use transformerlens. This can be done by comparing the activations between the clean prompt run and the conflict prompt run. One thing to do here is train a classifier (logistic regression) on what is conflict or not.

Details:

- We use gemma 2 and transformerlens to get the activations.
- We concatenate the prompt, question, and answer options into a single string. Then concatenate this with asking the model to output the answer.
- With the dataset and model in hand, we next examine the model’s internal activations to see if intra-context conflicts are detectable. We run each prompt through the model and capture activations at various points (e.g. at each layer’s output, or the residual stream after each token).
- We will start with probing the final layer’s residual stream at the answer position.
- Then we compare between the activation layers to see if (1) we are able to train a classifier to detect conflict (i want to use logistic regression) and (2) if we are able to retrieve the vector for the conflict.

### Checking causality

Check causality. We can add a fourth option to a neutral question, such as 'unsure' and increase this vector, and see how the model responds. We can also check the entropy of the output tokens. We can also just change the question to 'is there a contradiction in the prompt?' and see how the model responds based on how we change this vector.

To do this, we first need to retrieve the vector direction of 'conflict'. To do this, we retrieve the weights of our classifier at each layer.

Next, we want to test for causality. There are two ways to do this:

1. Add a fourth option to a neutral question, such as 'unsure' and increase this vector, and see how the model responds.
2. Change the question to 'is there a contradiction in the prompt?' and see how the model responds based on how we change this vector.

Let's write up both.

### Evaluating the results

Let's then evaluate the results up to this point, and iterate further on how we go from there.

### Immediate Task:

Here is what I want to do. I want to generate the dataset first. Write me a function/python script to do so where I can specify the max number of examples. Store the examples in a JSON file.

We should generate the dataset in a two-step manner:

1. Create the pairs of clean / conflict statements. I should probably want to generate these all together, I want diverse different statements? but if this is too difficult we can change this.
2. Parse these into a list of statements pairs with their metadata (like category).
3. Also generate multiple choice questions included with this, which should directly reference the contradiction in question
4. Store the question as a separate field in each 'data record' as well.

I think at this point, the data record should have:

- clean prompt, conflict prompt, question + multiple choice answers, category

I want to use gpt-5 mini. I already have my API key as an environment variable.
