# Portfolio wording and interview notes

## CV / project entry
**Nuclei segmentation and image-quality robustness | Python, scikit-image, SciPy, Streamlit**

Portfolio prototype comparing connected-components and watershed segmentation on annotated fluorescence microscopy images. Includes validation-based parameter selection, held-out instance matching, reproducible image-quality stress tests, morphology measurements and an interactive demo.

## Ownership and disclosure
This repository was created with AI assistance for Fabio's portfolio. It is not evidence of prior employment, clinical deployment, independent implementation or neural-network training. Describe your actual contribution when presenting it. Do not claim to have written or validated components you have not reviewed.

## Five questions to be able to answer
1. Why count nuclei rather than cells? The dataset's DNA stain exposes nuclei; whole-cell boundaries cannot be inferred reliably from that channel alone.
2. Why compare two simple methods? Connected components supplies a low-complexity baseline; watershed attempts to resolve touching regions.
3. Why do counts and Dice disagree? A merged region can cover the right pixels but correspond to several reference nuclei.
4. Why does brightness scaling have little effect? Percentile normalization cancels uniform scaling; this is not equivalent to lower exposure with noise.
5. What makes the results credible? Official splits, validation-only selection, archived per-image outputs, explicit failure examples and no invented deep-learning claims.

## How to discuss a weak result
State the actual metric and explain the failure mode. A portfolio experiment is valuable when it exposes limitations and suggests a well-designed next experiment. Do not present a classical baseline as a production medical system.
