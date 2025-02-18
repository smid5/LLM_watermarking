# SimMark: A SimHash Watermarking Scheme for Language Models

## Quick Start

1. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

2. Run the experiments using the following command:
```bash
python run.py
```

## SimMark File Structure

```
.
---data
    |---prompts.txt
    |---{generation_name}_{detection_namel}_{attack_name}.txt
---figures
    |---{plot_type}_{generation_name}_{num_tokens}.png
---simmark
    |---experiments
        |---utils.py # Shared functions for experiments
        |---attacks.py # Attack functions
        |---{experiment}.py
    |---methods
        |---expmin.py
        |---nomark.py
        |---redgreen.py
        |---simmark.py
```

# Watermark Detection Experiments

This section contains information three Python scripts for analyzing the statistical properties of watermarked text using p-values. These experiments explore how modifications, sentence length, and different watermarking parameters affect p-value distributions.

## Files and Descriptions

### 1. `num_modifications_vs_p_value.py`
- **Purpose:** Analyzes the effect of text modifications on SimHash p-values.
- **Key Functions:**
  - `plot_p_value_modifications()`: Plots the number of modifications against average p-values.
  - `generate_p_value_modification_experiment()`: Runs the experiment by modifying text iteratively and testing the p-value.
- **Output:** Saves a plot showing how modifications impact SimHash p-values.

### 2. `sentence_length_vs_p_value.py`
- **Purpose:** Examines how sentence length affects the median p-value under different watermarking conditions.
- **Key Functions:**
  - `plot_sentence_length_p_values()`: Generates a plot of sentence length versus median p-values.
  - `generate_sentence_length_p_values()`: Runs experiments with different sentence lengths and compares p-values for:
    - No watermark
    - SimMark watermark
    - SimMark with an attack
- **Output:** Saves a plot displaying the relationship between sentence length and p-values.

### 3. `k_and_b_heatmaps_pvalue.py`
- **Purpose:** Generates heatmaps to analyze how different `k` and `b` values impact watermark detection p-values.
- **Key Functions:**
  - `plot_heatmap()`: Creates heatmaps to visualize p-values for different `k` and `b` values.
  - `generate_p_value_heatmaps()`: Runs experiments on various `(k, b)` pairs to determine their influence on p-values.
- **Output:** Saves heatmaps illustrating p-value distributions for:
  - Watermarked text
  - Unrelated text
  - Attacked text

