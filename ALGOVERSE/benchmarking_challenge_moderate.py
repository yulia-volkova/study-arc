# %% [markdown]
# # LLM Benchmark Comparison: GPT-4o-mini vs GPT-4.1-nano
# This script runs a fixed benchmark evaluation for two models,
# and generates a summary plot by behavior category.

# %% [markdown]
# ## 1. Imports and Setup

# %%
from inspect_ai import eval
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver
from inspect_ai.solver import chain, generate
from inspect_ai.scorer import answer
from inspect_ai import Task, eval, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import match, model_graded_fact
from inspect_ai.solver import chain_of_thought, generate, self_critique
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import answer
from inspect_ai.solver import Choices

# %% [markdown]
# ## 2. Define Paths and Models

# %%
models = ["gpt-4o-mini", "gpt-4.1-nano"]
log_base_dir = Path.home() / "ARENA_3.0" / "logs" / "final_eval_comparison"
log_base_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 3. ARC Dataset Loader

# %%
from inspect_ai.dataset import Sample, hf_dataset


# %%
import re
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser

def classify_arc_behavior(input_text: str) -> str:
    text = input_text.lower()
    if "cause" in text or "effect" in text:
        return "causal"
    if "not" in text or "except" in text:
        return "negation"
    if "compare" in text or "difference" in text:
        return "comparison"
    return "recall"

def arc_record_to_sample(record: dict) -> Sample:
    labels = record["choices"]["label"]
    choices = record["choices"]["text"]
    target = chr(ord("A") + labels.index(record["answerKey"]))
    input_text = record["question"]

    category = classify_arc_behavior(input_text)

    return Sample(
        input=[ChatMessageUser(content=input_text)],
        choices=choices,
        target=target,
        metadata={"behavior_category": category}
    )



# %%
# Load the ARC dataset
from datasets import load_dataset

arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")

# %%
# Pick and inspect one record
raw = arc[10]
sample = arc_record_to_sample(raw)

# %%
# View its full contents
print("=== Sample Inspection ===")
print("Prompt:", sample.input[0].content)
print("Choices:", sample.choices)
print("Correct Answer:", sample.target)
print("Behavior Category:", sample.metadata["behavior_category"])

# %%
# Helper functions

def letters_and_answer_options(choices: Choices) -> tuple[str, str]:
    """
    Helper function, returns `choices` formatted as MCQ options, as well as the string of labels for each option.
    """
    letters = [chr(65 + i) for i in range(len(choices))]

    return (
        ", ".join(letters),
        "\n".join([f"{letter}) {choice.value}" for letter, choice in zip(letters, choices)]),
    )
# %%
# Templates

TEMPLATE_MCQ = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTERS' (without quotes) where LETTERS is one or more of {letters}.

{question}

{choices}"""

TEMPLATE_MCQ_COT = r"""Think about the following question, without coming to a final answer:

{question}

{choices}"""

TEMPLATE_MCQ_MAKE_CHOICE = r"""Please make a choice from the options above. 
    
Your answer should be in the form 'ANSWER: $LETTERS' (without quotes) where LETTERS is one or more of {letters}."""

TEMPLATE_MCQ_COT_AND_CHOICE = r"""Think about the following question:

{question}

{choices}

Before answering, think through step-by-step how to answer the question. Make sure to think step-by-step and weigh all relevant concerns, and then arrive at a conclusive answer. Your final answer should be in the form 'ANSWER: $LETTERS' (without quotes) where LETTERS is one or more of {letters}."""
# %%
# helper solvers
@solver
def make_choice(prompt: str = TEMPLATE_MCQ_MAKE_CHOICE) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        letters = letters_and_answer_options(state.choices)[0]
        chat_history = state.messages
        input = prompt.format(letters = letters)
        chat_history.append(ChatMessageUser(content = input))
        state.messages = chat_history
        return state

    return solve

@solver
def multiple_choice_format(template: str = TEMPLATE_MCQ) -> Solver:
    """
    Returns a solve function which modifies the initial prompt to be in the format of an MCQ.
    """
    tags = set(re.findall(r"\{.*?\}", template))
    assert r"{question}" in tags, "Template must include {question} field"
    assert r"{choices}" in tags, "Template must include {choices} field"
    assert tags - {r"{question}", r"{choices}", r"{letters}"} == set(), "Unexpected field found in template"

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        assert state.choices, "If using MCQ then state must have `choices` field"
        letters, answer_options = letters_and_answer_options(state.choices)
        question = state.user_prompt.text
        state.user_prompt.text = template.format(question=question, choices = answer_options, letters = letters )

        return state

    return solve


# %% [markdown]
# ## 4. Eval Task Definition for ARC Dataset

# %%
@task
def arc_eval(n: int | None, use_cot=True, use_self_critique=True, self_critique_model=None) -> Task:
    dataset = hf_dataset(
        path="allenai/ai2_arc",
        name="ARC-Challenge",
        sample_fields=arc_record_to_sample,
        split="validation",
        trust=True,
        limit=n,
    )
    steps = []
    if use_cot:
        steps.append(multiple_choice_format())
        steps.append(generate())
        if use_self_critique:
            steps.append(self_critique(model=self_critique_model))
            steps.append(generate())
            steps.append(make_choice())
            steps.append(generate())
    else:
        steps.append(multiple_choice_format())
        steps.append(generate())
    return Task(dataset=dataset, solver=chain(*steps), scorer=answer("letter"))

# %% [markdown]
# ## 5. Run Evaluation for Each Model

# %%
for model in models:
    sample_no = 299
    task_name = f"arc-system_cot-True_critique-True_{model}_samples_{sample_no}"
    log_dir = log_base_dir / task_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Running evaluation for: {model}")

    logs = eval(
        arc_eval(
            n=sample_no,
            use_cot=True,
            use_self_critique=True,
            self_critique_model=f"openai/{model}",
        ),
        model=f"openai/{model}",
        log_dir=str(log_dir),
    )

# %% [markdown]
# ## 6. Paired Analysis: Direct Comparison Between Models
# %%
from inspect_ai.analysis.beta import samples_df
from pathlib import Path
import pandas as pd

log_base_dir = Path.home() / "ARENA_3.0" / "logs" / "final_eval_comparison" 

df = samples_df(logs=log_base_dir)

print(df.columns)
df["correct"] = df["score_answer"].map({"C": 1, "I": 0})

def extract_model_name(usage_str: str) -> str:
    try:
        data = json.loads(usage_str)
        return next(iter(data.keys()))
    except Exception:
        return "unknown"


df["model_name"] = df["model_usage"].apply(extract_model_name)
print(df["score_answer"].unique())
# Count occurrences of each score 
count_summary = (
    df
    .groupby(["model_name", "score_answer"])
    .size()
    .unstack(fill_value=0)
)
print(count_summary)


count_table = df.groupby(["model_name", "score_answer"]).size().unstack(fill_value=0)
print("Count of Correct (C) and Incorrect (I) answers per model:\n", count_table)


# %%
from IPython.display import display

combined = df.pivot_table(
    index=["input",  "target"],
    columns="model_name",
    values="score_answer",
    aggfunc="first"
).reset_index()


model_cols = [c for c in combined.columns if c not in ("input", "target")]
mask = combined[model_cols].isin(["I"]).any(axis=1)
wrong_questions = combined[mask]

# Show results
print(f"Total inputs with at least one incorrect answer: {len(wrong_questions)}")
display(wrong_questions)


# %%
mA = "openai/gpt-4.1-nano"
mB = "openai/gpt-4o-mini"

# Filter incorrect answers
wrong_A = combined[combined[mA] == "I"]
wrong_B = combined[combined[mB] == "I"]
wrong_both = combined[(combined[mA] == "I") & (combined[mB] == "I")]

print(f"GPT-4.1-nano wrong: {len(wrong_A)}")
print(f"GPT-4o-mini wrong: {len(wrong_B)}")
print(f"Both models wrong: {len(wrong_both)}")

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Style TF-IDF setup
vectorizer = TfidfVectorizer(
    stop_words="english",
    min_df=2,
    max_df=0.8,
    ngram_range=(1,2)
)

def top_tfidf_terms(texts, top_n=10):
    tfidf = vectorizer.fit_transform(texts)
    df_tfidf = pd.DataFrame(
        tfidf.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    return df_tfidf.sum(axis=0).nlargest(top_n).index.tolist()

# Extract top terms
print("Top terms GPT‑4.1‑nano missed:", top_tfidf_terms(wrong_A["input"]))
print("Top terms GPT‑4o‑mini missed:", top_tfidf_terms(wrong_B["input"]))
print("Top terms both missed:", top_tfidf_terms(wrong_both["input"]))


# %% [markdown]

# I am seeing that there is at least some intersection in the terms that both models find difficult:

# Top terms GPT‑4.1‑nano missed: ['likely', 'best', 'following', 'environment', 'user best', 'resources', 'user following', 'plants', 'energy', 'earth']
# Top terms GPT‑4o‑mini missed: ['best', 'likely', 'following', 'science', 'energy', 'natural', 'survive', 'user best', 'species', 'environment']
# Top terms both missed: ['best', 'likely', 'survive', 'energy', 'user best']
# %%

nano = "openai/gpt-4.1-nano"
mini = "openai/gpt-4o-mini"

# Define masks
only_nano_wrong = combined[(combined[nano] == "I") & (combined[mini] != "I")]
only_mini_wrong = combined[(combined[mini] == "I") & (combined[nano] != "I")]
both_wrong = combined[(combined[nano] == "I") & (combined[mini] == "I")]

# Helper for displaying examples
def show_examples(df_subset, label, n=3):
    print(f"\n=== {label} (n={len(df_subset)}) ===")
    if df_subset.empty:
        print("No examples found.")
        return
    samples = df_subset.sample(min(n, len(df_subset)), random_state=42)
    display(samples[["input", nano, mini]])

# Show 3 examples each
show_examples(only_nano_wrong, f"Only GPT‑4.1‑nano wrong", 3)
show_examples(only_mini_wrong, f"Only GPT‑4o‑mini wrong", 3)
show_examples(both_wrong,           "Both models wrong", 3)

# %%

all_wrong = pd.concat([only_nano_wrong, only_mini_wrong, both_wrong], ignore_index=True)

# Display example
from IPython.display import display
print(f"Total wrong cases combined: {len(all_wrong)}")
display(all_wrong.sample(min(5, len(all_wrong)), random_state=42))
# %%
# %%
out_path = Path.home() / "ARENA_3.0/logs/final_eval_comparison/result.csv"
all_wrong.to_csv(out_path, index=False)
print("All wrong answers exported to:", out_path)

# %%
nano, mini = "openai/gpt-4.1-nano", "openai/gpt-4o-mini"

combined["both_correct"] = (combined[nano] == "C") & (combined[mini] == "C")
combined["both_wrong"]   = (combined[nano] == "I") & (combined[mini] == "I")

# %%
combined["length"] = combined["input"].str.split().apply(len)

length_correct = combined.loc[combined["both_correct"], "length"]
length_wrong   = combined.loc[combined["both_wrong"], "length"]

print(f"Mean length (both correct): {length_correct.mean():.1f} words")
print(f"Mean length (both wrong):   {length_wrong.mean():.1f} words")

# %%
from scipy.stats import ttest_ind

t_stat, p_val = ttest_ind(length_correct, length_wrong, equal_var=False)
print(f"T-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
# %% [markdown]
# I looked at length differences of questions that both models get correct and both get worng 
# Mean length (both correct): 25.2 words
# Mean length (both wrong):   20.2 words
# %% [markdown]
# T-statistic = 1.648, p-value = 0.121
# p value is not statistically significant
# %%
