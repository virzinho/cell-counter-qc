# Methods and evaluation protocol

## Task
Instance segmentation and counting of nuclei in single-channel 2D fluorescence microscopy. The project does not identify diseases or infer viability, cell type or treatment response. Whole-cell boundaries are not available in the DNA channel.

## Dataset
BBBC039v1 contains 200 fields of view from a chemical screen on U2OS cells. Images are 16-bit TIFF, 520 × 696 pixels. The DNA channel is accompanied by manually annotated nucleus masks and official training, validation and test lists. Dataset and annotations are CC0. See https://bbbc.broadinstitute.org/BBBC039.

Ground truth decoding follows the approach linked by Broad: take the first PNG channel and label connected regions of equal intensity. Labeling the entire foreground as a binary mask would merge touching nuclei and invalidate instance counts. Each nucleus touching the image border is retained in both reference and prediction.

## Pipeline
1. Validate dimensions and finite values; retain 16-bit dynamic range before conversion to float.
2. Scale using image intensity percentiles 1 and 99.8, clipping to [0, 1]. For sparse images with coinciding percentiles, use min/max. Constant images return an empty mask.
3. Gaussian smoothing with sigma = 1 pixel.
4. Foreground threshold: Otsu × multiplier. Remove small objects, then fill holes.
5. Connected-components baseline: each component is one nucleus.
6. Watershed baseline: Euclidean distance transform, spaced local maxima as markers, and watershed of the negative distance map restricted to the foreground.
7. Remove undersized output instances and renumber labels. Export area, centroid, equivalent diameter, eccentricity and solidity.

No neural network is trained. There is no claim of state-of-the-art performance.

## Parameter selection
The candidate grid is min_area ∈ {40, 80}, threshold_factor ∈ {0.7, 1.0}, min_distance ∈ {6, 10}. Connected components ignore peak distance and have four candidate configurations; watershed has eight. All candidates use sigma = 1.

Choose each method's configuration by maximum mean per-image instance F1 on the official validation split; break ties using minimum count MAE. The official training split is reserved and not used because neither method requires training. The test split is used only after parameter selection. See `reports/validation_search.csv` and `reports/selected_config.json`.

## Metrics
- Count MAE: mean absolute difference between predicted and reference nucleus counts per image.
- Instance F1: one-to-one matching of predicted and reference regions at IoU ≥ 0.5. Hungarian assignment maximizes valid match count, with IoU as a secondary criterion. False splits and merges reduce this score.
- Foreground Dice: pixel overlap after discarding instance identity. A merged prediction can have perfect Dice and still count incorrectly, so Dice is not enough.
- Primary F1 is a macro mean over images. Micro F1, pooling TP/FP/FN, is included in the summary CSV.
- 95% confidence intervals use 2,000 bootstrap resamples of images (seed 42). They describe this split, not external generalization, and ignore correlations between biological images.

## Stress tests
Original, Gaussian blur sigma 2 and 4, clipped Gaussian noise sigma .08, uniform intensity scaling ×.4, and blur sigma 2 plus noise sigma .08. Perturbations are applied to normalized image intensities; fixed image-specific seeds derive from SHA-256 of the filename. The same altered image is passed to both methods.

Uniform brightness scaling is a control: the pipeline's normalization largely cancels it. It does not simulate low-exposure shot noise or quantization. Likewise, digital Gaussian blur is not a full model of optical defocus. Masks are unchanged: the experiment measures robustness against the original reference objects, not annotator visibility under corruption.

## Limitations
- One cell line, one screen, one imaging modality. No validation on another laboratory or microscope.
- Shape and density variation can cause over-segmentation or merging. Weak signals can disappear at thresholding.
- Quality indicators are not calibrated rejection criteria. Noise can increase apparent sharpness.
- Morphology uses pixel units. No microscope pixel-size calibration is supplied.
- The small test set limits precision. Threshold sensitivity and per-image errors are reported rather than hidden.
- AI-assisted implementation; code and protocol should be reviewed before scientific reuse.

## Extensions
Evaluate a pretrained or newly trained U-Net/Cellpose baseline using appropriate licenses; add an external dataset; calibrate quality rejection on independent examples; compare Fiji workflows; include image-scale metadata. These are future work, not delivered results.
