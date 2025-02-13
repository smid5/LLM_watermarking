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

## Notes

- When using the `test_watermark` method in the `simmark/experiments/utils.py` file, outputs are automatically cached in the `data` directory.

