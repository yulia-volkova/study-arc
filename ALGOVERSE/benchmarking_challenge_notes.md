# LLM Benchmark Summary: GPT‑4o‑mini vs GPT‑4.1‑nano

## Evaluation Setup
- Compared `openai/gpt-4o-mini` and `openai/gpt-4.1-nano` on the **ARC Challenge** dataset using 299 samples.
- Evaluation used `inspect_ai` with:
  - Chain-of-thought reasoning
  - Self-critique step
  - Multiple-choice formatted questions
- Created custom behavior categories (e.g., causal, negation, comparison, recall), but these did not yield useful differentiation, almost all incorrectly answered questions were 'recall' category. 

## Accuracy Results
- GPT-4o-mini: **278 correct**, 21 incorrect → **Accuracy ~93%**
- GPT-4.1-nano: **270 correct**, 29 incorrect → **Accuracy ~90.3%**

## Paired Model Comparison
Count of Correct (C) and Incorrect (I) Answers per Model:
    
    model_name            C     I
    openai/gpt-4.1-nano   270   29
    openai/gpt-4o-mini    278   21

Wrong answer stats:
    
    GPT-4.1-nano wrong: 29
    GPT-4o-mini   wrong: 21
    Both models   wrong: 12

## Error Example Exploration
- Reviewed randomly sampled inputs that only one model or both models got wrong.
- Exported all incorrect cases to CSV for deeper manual analysis.
- Did not notice any patterns looking by eye

## TF-IDF Analysis of Missed Questions
- **Top terms GPT-4.1-nano missed**: likely, best, following, environment, user best, resources, plants, energy, earth
- **Top terms GPT-4o-mini missed**: best, likely, following, science, energy, natural, survive, species, environment
- **Common hard terms**: best, likely, survive, energy, user best

Insight: Both models tend to struggle with ambiguous or general reasoning questions using terms like "best", "likely", and "survive".

## Length Analysis
Compared word lengths of questions where both models were correct vs. both were wrong:

- Mean length (both correct): **25.2 words**
- Mean length (both wrong): **20.2 words**

T-test results:
- T-statistic = 1.648
- p-value = 0.121 → **not statistically significant**

## Observations & Thoughts
- GPT-4o-mini slightly outperforms GPT-4.1-nano on ARC Challenge tasks with ~3% higher accuracy.
- Although both models share weaknesses, GPT-4o-mini appears more robust on ambiguous or nuanced questions such as 
    - "Although they belong to the same family, an eagle and a pelican are different. What is one difference between them?"
- The custom behavior categories did not yield meaningful trends — possible that more granular taxonomy or smarter classification will be effective, my classification was very simple.
- I analysed average lentgh of correctly and incorrectly answered questions. Length alone doesn't explain model failure — no statistically significant difference between lengths of correctly and incorrectly answered prompts. However, my sample may be too small. 
- I found that the questions that both models fail to answer have same terms as most frequent terms by tf-idf.  

