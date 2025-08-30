import numpy as np
from .utils import load_llm_config, load_prompts, test_watermark, METHODS

def compute_tpr(pvals_unwm, pvals_wm, fpr):
    # Step 1: Find threshold tau from unwatermarked p-values
    sorted_unwm = np.sort(pvals_unwm)
    print(len(sorted_unwm))
    k = int(np.floor(fpr * len(sorted_unwm)))
    tau = sorted_unwm[k]

    # Step 2: Compute TPR using the same threshold
    tpr = np.mean(pvals_wm <= tau)
    return tpr, tau

def get_tprs(method_name, fprs, num_tokens, filename, k=4, b=4, seeds=[42]):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)

    method = f"simmark_{k}_{b}" if method_name == "SimMark" else METHODS[method_name]
    detection_name = method

    # Generate without watermark
    pvals_unwm = np.concatenate([test_watermark(prompts, num_tokens, llm_config, "nomark", detection_name, seed=seed) 
                  for seed in seeds])

    # Generate with watermark
    pvals_wm = np.concatenate([test_watermark(prompts, num_tokens, llm_config, method, detection_name, seed=seed)
                for seed in seeds])

    for fpr in fprs:
        tpr, tau = compute_tpr(pvals_unwm, pvals_wm, fpr)
        print(f"Method: {method_name}, FPR: {fpr}, TPR: {tpr}, Tau: {tau}")