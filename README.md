# zeze
Personal website
# Biophysical Parameter-to-eFEL Neural Network Surrogate

This project builds a neural-network surrogate for neuronal simulation data.
The model learns the following relationship:

```text
biophysical parameters + stimulation amplitude
                    ↓
        predicted eFEL features
```

The project has four main stages:

1. Read and clean MOO generation CSV files.
2. transform and normalize the data.
3. Train separate neural-network regressors for no-spike and spiking responses.
4. Use model gradients to estimate which input parameters are most important for each eFEL feature.

The current implementation is designed for a large dataset that may not fit fully in memory. It uses per-neuron caches, memory-mapped arrays, optional in-RAM training subsets, masked loss, early stopping, and regime-specific models.

---

## 1. Main Files

| File | Purpose |
|---|---|
| `data_analysis.py` | Creates before-cleaning and after-cleaning statistics, quantiles, and parameter-range CSV reports. |
| `data_pipline.py` | Reads CSV and JSON files, builds caches, applies cleaning, log transforms, masks, and normalization, and returns a PyTorch dataset. |
| `multi_model.py` | Defines the neural-network models, training loops, evaluation metrics, checkpoint saving, and plots. |
| `sensitivity_analysis.py` | Loads trained regime models and calculates Mean Absolute Gradient sensitivity scores. |
| `plot_sensitivity.py` | Converts sensitivity scores into row-normalized heatmaps. |

> Note: The filename is currently written as `data_pipline.py`, not `data_pipeline.py`. Imports must use the same spelling unless the file is renamed everywhere.

---

## 2. End-to-End Workflow

```text
Generation CSV files
        +
protocol_stage JSON files
        │
        ▼
Read parameters, protocol features, and stimulus amplitudes
        │
        ▼
Create raw per-neuron cache files
        │
        ▼
Create real .npy memory maps for row-by-row reading
        │
        ▼
Apply feature validity rules
        │
        ├── bounds checking
        ├── optional signed log transform
        ├── Z-score normalization
        └── masks for unavailable or invalid targets
        │
        ▼
Split samples by Spikecount
        │
        ├── class 0: no_spike
        └── class 1: spiking
        │
        ▼
Train one residual MLP regressor per response class
        │
        ▼
Evaluate with validation loss, R², MAE, and parity plots
        │
        ▼
Calculate gradients of each output with respect to each input
        │
        ▼
Average absolute gradients and rank parameter importance
```

---

# Part A — Data Analysis and Filtering

## 3. Expected Input Data

### 3.1 Generation CSV files

The code searches recursively for files with names matching:

```text
*_gen_*.csv
```

The input path may be:

- one neuron folder, or
- a parent folder containing several neuron folders.

The code sorts files first by neuron folder and then by generation number.

### 3.2 Column naming rules

Columns are classified by the number of dot-separated parts in the column name.

```text
parameter.zone
protocol.location.feature
```

The rules are:

- Two parts, such as `gbar_SK.somatic`, are treated as input parameters.
- Three parts, such as `Step1.soma.AP1_peak`, are treated as target features.
- Columns beginning with `_` are ignored.

The final parameter list is discovered from the CSV header. It is not hard-coded.

### 3.3 Protocol amplitudes

Stimulation amplitudes are read from files matching:

```text
*protocol_stage*.json
```

For every protocol ID, the code reads:

```python
value["stimuli"][0]["amp"]
```

The stimulation amplitude is added as one extra model input.

Therefore:

```text
input dimension = number of biophysical parameters + 1 amplitude value
```

---

## 4. Target eFEL Features

The pipeline uses 14 canonical target features:

1. `AHP1_depth_from_peak`
2. `AHP_depth`
3. `AHP_time_from_peak`
4. `AP1_peak`
5. `AP1_width`
6. `Spikecount`
7. `decay_time_constant_after_stim`
8. `inv_first_ISI`
9. `mean_AP_amplitude`
10. `sag_amplitude`
11. `steady_state_voltage`
12. `steady_state_voltage_stimend`
13. `time_to_first_spike`
14. `voltage_base`

Each protocol is expanded into a separate training block. The same parameter vector is repeated for each protocol, while the protocol amplitude and available target features change.

---

## 5. Missing Values and Masks

Not every eFEL feature is available for every protocol or every response.

For every sample, the pipeline creates:

- a target vector, `features`, and
- a validity vector, `masks`.

A mask value has the following meaning:

```text
1 = this target value is valid and may be used in the loss
0 = this target value is missing, unavailable, or invalid
```

Missing target values are stored as zero in the target array, but the zero is ignored because the corresponding mask is also zero.

This allows one model to predict several outputs even when some outputs are not defined for every sample.

---

## 6. Data Types and Memory Use

CSV numeric values are loaded as `float32` when possible. This uses about half the memory of `float64`.

The pipeline also avoids one very large global in-memory array. It creates one raw cache file per neuron:

```text
.moo_mmap_cache/raw_track_<neuron_id>.npz
```

Each NPZ cache is later unpacked once into real `.npy` files:

```text
.moo_mmap_cache/mm_raw_track_<neuron_id>/
    params.npy
    amplitudes.npy
    features.npy
    masks.npy
```

These `.npy` files are opened with `mmap_mode="r"`. The dataset can then read individual rows without loading the full dataset into RAM.

---

## 7. Cleaning Rules Used by the Training Dataset

The training dataset uses lower and upper physiological bounds.

### 7.1 Upper bounds

| Feature | Upper bound |
|---|---:|
| `decay_time_constant_after_stim` | 500.0 |
| `AP1_width` | 15.0 ms |
| `inv_first_ISI` | 500.0 Hz |
| `AHP_time_from_peak` | 1000.0 ms |
| `time_to_first_spike` | 1500.0 ms |
| `AHP1_depth_from_peak` | 200.0 mV |
| `AHP_depth` | 150.0 mV |
| `mean_AP_amplitude` | 200.0 mV |
| `sag_amplitude` | 7.0 mV |

### 7.2 Lower bounds

The following features must be at least zero:

- `decay_time_constant_after_stim`
- `AP1_width`
- `AHP1_depth_from_peak`
- `AHP_depth`
- `mean_AP_amplitude`
- `sag_amplitude`
- `time_to_first_spike`
- `AP1_peak`

Voltage features such as `voltage_base` and `steady_state_voltage` are allowed to be negative.

### 7.3 Current row-wise invalidation behavior

Inside `MOODataset.__getitem__`, the code checks all valid target cells in a sample.

If one checked feature is outside its allowed range, the code does this:

```python
mask[:] = 0.0
```

This means the complete sample row becomes invalid for every target feature.

Example:

```text
A row contains a sag_amplitude of 76.54 mV.
The upper bound is 7.0 mV.
The complete target mask for that row becomes zero.
```

This is stricter than removing only the bad target cell. It can strongly reduce the amount of valid data for other features in the same row.

This behavior helps explain why `sag_amplitude` can have too little valid data in the no-spike evaluation.

---

## 8. Cleaning Rules Used by `data_analysis.py`

The reporting script performs two passes.

### Pass 1 — Before cleaning

It computes statistics with:

- raw values,
- optional log configuration,
- no negative-value filtering,
- no report-specific upper-bound filtering.

It writes a detailed before-cleaning CSV.

### Pass 2 — After cleaning

It applies row-wise invalidation based on the report configuration.

Negative values are rejected for:

- `time_to_first_spike`
- `AHP1_depth_from_peak`
- `AHP_depth`
- `AP1_width`
- `AP1_peak`
- `mean_AP_amplitude`
- `sag_amplitude`

The report-specific upper bounds are:

| Feature | Upper bound |
|---|---:|
| `AP1_width` | 15.0 |
| `inv_first_ISI` | 500.0 |
| `AHP_time_from_peak` | 1000.0 |
| `time_to_first_spike` | 1500.0 |

The script calculates pooled values across all neurons:

- number of valid values,
- negative, zero, and positive counts,
- minimum and maximum,
- mean and standard deviation,
- Q01, Q05, Q25, median, Q75, Q95, and Q99,
- positive-only quantiles.

All reported statistics are returned to raw physiological units.

### Important difference

The report script and the training dataset do not currently use exactly the same bound dictionaries.

For example, the training dataset also checks bounds for `sag_amplitude`, `AHP_depth`, and several other features. The report script's local `UPPER_BOUNDS` dictionary contains only four features.

For a fully matched report, both scripts should use one shared cleaning configuration.

---

## 9. Log Transforms

The pipeline currently applies a signed `log1p` transform to:

- `decay_time_constant_after_stim`
- `AHP_time_from_peak`
- `time_to_first_spike`
- `inv_first_ISI`

The formula is:

```text
signed_log1p(x) = sign(x) × log(1 + |x|)
```

`AP1_width` is currently not log-transformed. It appears as a commented line in `LOG_TRANSFORM`.

The signed form can safely represent zero and negative values, although the current cleaning rules remove negative values for several features before training.

For evaluation, predictions are converted back with:

```text
signed_expm1(x) = sign(x) × (exp(|x|) - 1)
```

---

## 10. Z-Score Normalization

Inputs and valid targets are normalized with:

```text
z = (value - mean) / standard_deviation
```

The following statistics are stored:

- parameter mean and standard deviation,
- amplitude mean and standard deviation,
- one mean and standard deviation for each target feature.

Feature statistics are calculated only from cells whose mask is valid.

After normalization, invalid target cells are forced to zero:

```python
features = features * masks
```

### Current implementation detail

The streaming normalization-statistics function removes out-of-bound feature cells one cell at a time while calculating means and standard deviations.

The final `__getitem__` path invalidates the full row when one bounded feature is bad.

Therefore, normalization statistics and final training masks are not based on exactly the same invalidation rule. This should be aligned in a future cleanup for complete consistency.

---

# Part B — Response Regimes

## 11. No-Spike and Spiking Classes

The code finds the `Spikecount` feature and rounds it to an integer.

The class rule is:

```text
Spikecount < 1  → class 0 → no_spike
Spikecount ≥ 1  → class 1 → spiking
```

Only samples with a valid `Spikecount` mask are used for class selection.

### 11.1 No-spike regressor outputs

The no-spike model is trained on five selected targets:

- `voltage_base`
- `steady_state_voltage`
- `steady_state_voltage_stimend`
- `sag_amplitude`
- `decay_time_constant_after_stim`

### 11.2 Spiking regressor outputs

The spiking model is trained on 12 selected targets:

- `AHP1_depth_from_peak`
- `AHP_depth`
- `AHP_time_from_peak`
- `AP1_peak`
- `AP1_width`
- `mean_AP_amplitude`
- `time_to_first_spike`
- `steady_state_voltage_stimend`
- `steady_state_voltage`
- `voltage_base`
- `Spikecount`
- `inv_first_ISI`

The feature names are sorted before their indices are sent to the model. The selected indices saved in the checkpoint are therefore the most reliable record of output order.

---

# Part C — Neural-Network Structure

## 12. Residual Regressor Architecture

The main regression model is `FeaturePredictor`.

Its actual default structure is:

```text
Input: all parameters + amplitude
        │
        ▼
Linear(input_dim → 256)
ReLU
        │
        ▼
Residual Block 1
    Linear(256 → 256)
    ReLU
    Dropout(0.10)
    Linear(256 → 256)
    Add skip connection
    ReLU
        │
        ▼
Residual Block 2
    Linear(256 → 256)
    ReLU
    Dropout(0.10)
    Linear(256 → 256)
    Add skip connection
    ReLU
        │
        ▼
Linear(256 → number of selected outputs)
```

A residual block calculates:

```text
output = ReLU(input + block(input))
```

The skip connection gives gradients a shorter path through the model. This can make a deeper model easier to train.

### Important architecture detail

The function default is written as:

```python
hidden_dims=(256, 128, 64)
```

However, `FeaturePredictor` currently uses only the first value:

```python
dim = hidden_dims[0]
```

Therefore, the active regressor width is 256 throughout both residual blocks. It is not a 256 → 128 → 64 regressor.

The optional `SpikeClassifier` still uses all three widths, 256 → 128 → 64, in a normal feed-forward MLP.

### About claimed improvements

The current files show the new residual architecture and the current R² scores. They do not include the older model's matched validation metrics. Therefore, an exact statement such as “R² improved by more than 52%” cannot be verified from these files alone.

To report an improvement percentage, save before-and-after metrics using the same data split, seed, cleaning rules, and evaluation code.

---

## 13. Model Inputs and Outputs

### Inputs

The model input is:

```text
[parameter_1, parameter_2, ..., parameter_n, amplitude]
```

All input values are Z-score normalized.

### Outputs

Each regime has its own output dimension:

```text
no_spike output dimension = 5
spiking output dimension = 12
```

Outputs are predicted in normalized target space during training.

---

# Part D — Training Procedure

## 14. Reproducibility

The global random seed is:

```text
42
```

The code sets seeds for:

- Python `random`,
- NumPy,
- PyTorch CPU,
- PyTorch CUDA,
- `PYTHONHASHSEED`,
- train-validation splits,
- DataLoader generators,
- class subsampling.

`FAST=True` is enabled. This allows faster backend behavior and sets `cudnn.benchmark=True` when CUDA is used. Because of this, CUDA runs are not guaranteed to be perfectly deterministic even though seeds are set.

---

## 15. Class Subsampling

The maximum number of samples used for each response class is:

```text
SUBSAMPLE_PER_CLASS = 1,000,000
```

When a class has more than one million samples, the code selects one million with Python's seeded random sampler.

The same selected class pool is then divided into training and validation data.

---

## 16. Train-Validation Split

The default validation fraction is:

```text
10% validation
90% training
```

The split uses seed 42.

For a one-million-sample class, this gives approximately:

```text
900,000 training samples
100,000 validation samples
```

---

## 17. In-RAM Training Mode

The default CLI behavior is:

```text
IN_RAM_DEFAULT = True
```

For each class, the selected subset is processed once through the normal dataset path and copied into a `TensorDataset` in RAM.

This means the following work is completed once before training:

- row reading from memory maps,
- bounds checking,
- log transform,
- Z-score normalization,
- target-mask preparation.

During later epochs, the training loop reads flat tensors from RAM instead of performing random reads from disk-backed memory maps.

### In-RAM batch size

```text
RAM_BATCH_SIZE = 8192
```

When in-RAM mode is active, this value replaces the normal training batch size.

### Memory-map batch size

The CLI default is:

```text
batch_size = 512
```

This remains the effective batch size when in-RAM mode is disabled.

### DataLoader workers

For disk-backed loading:

```text
NUM_WORKERS = 6
prefetch_factor = 4
persistent_workers = True
```

For in-RAM loading:

```text
num_workers = 0
```

This avoids worker-process communication overhead when the tensors are already in memory.

---

## 18. Masked Mean Squared Error

The regressors use masked MSE.

For prediction `ŷ`, target `y`, and mask `m`:

```text
Masked MSE = Σ[(ŷ - y)² × m] / max(Σm, 1)
```

Only valid target cells contribute to the loss.

This is important because no-spike samples do not have many spike-dependent measurements.

---

## 19. Optimizer, Scheduler, and Early Stopping

### Optimizer

```text
Adam
learning rate = 0.001
weight decay = 0.00001
```

The weight decay gives a small L2-style regularization effect.

### Learning-rate scheduler

```text
ReduceLROnPlateau
mode = minimize validation loss
factor = 0.5
scheduler patience = 5 epochs
```

If validation loss does not improve for five scheduler checks, the learning rate is multiplied by 0.5.

### Early stopping

```text
early-stopping patience = 15 epochs
maximum epochs = 200
```

The model stores the state with the lowest validation loss. When training stops, the best stored state is restored.

---

## 20. Full Default Hyperparameter Table

| Hyperparameter | Default | Meaning |
|---|---:|---|
| Global random seed | 42 | Controls splits, sampling, and initialization. |
| `SUBSAMPLE_PER_CLASS` | 1,000,000 | Maximum samples used from each response class. |
| Hidden width used by regressor | 256 | Width of both residual blocks. |
| Number of residual blocks | 2 | Two skip-connected blocks. |
| Dropout | 0.10 | Applied once inside each residual block. |
| Learning rate | 0.001 | Initial Adam learning rate. |
| Weight decay | 0.00001 | Adam regularization value. |
| Maximum epochs | 200 | Maximum full training passes. |
| Validation fraction | 0.10 | Ten percent of each class pool. |
| Early-stopping patience | 15 | Stops after 15 non-improving epochs. |
| Scheduler | ReduceLROnPlateau | Watches validation loss. |
| Scheduler factor | 0.5 | Halves the learning rate. |
| Scheduler patience | 5 | Waits five plateau epochs. |
| Standard batch size | 512 | Used for memory-map training. |
| In-RAM batch size | 8192 | Used when class data is materialized in RAM. |
| DataLoader workers | 6 | Used by disk-backed loaders. |
| DataLoader prefetch factor | 4 | Batches prefetched per worker. |
| `FAST` | `True` | Prioritizes throughput over strict determinism. |
| Drop fraction | 0.0 | No high-loss samples are removed by default. |
| Warmup epochs | 50 | Used only when the optional sample-dropping path is active. |
| Maximum sensitivity samples per regime | 50,000 | Current `__main__` sensitivity setting. |
| Sensitivity fallback pool | 300,000 | Random candidates examined when saved validation indices do not exist. |
| Sensitivity batch size | 4096 | Batch used during gradient analysis. |
| Plot parity maximum points | 5000 | Plotting only; R² uses all valid points. |

---

## 21. Optional High-Loss Sample Dropping

The code contains an optional experiment that:

1. trains a warmup model,
2. calculates one masked MSE value per sample,
3. removes a chosen fraction of the highest-loss samples,
4. retrains the final model.

The sweep values are:

```text
0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
```

This path is off by default.

In-RAM mode is also disabled automatically when `drop_fraction > 0`, because the current dropping code expects global dataset indices.

---

# Part E — Evaluation

## 22. Validation Metrics

Evaluation converts predictions back to raw physiological scale before calculating metrics.

### R²

```text
R² = 1 - Σ(y - ŷ)² / Σ(y - mean(y))²
```

Interpretation:

- `1.0` means nearly perfect predictions.
- `0.0` means the model is no better than predicting the target mean.
- A negative value means the model is worse than predicting the mean.

### MAE

```text
MAE = mean(|y - ŷ|)
```

MAE is reported in the target's original unit after denormalization and reverse log transform.

### Parity plot

Each parity plot compares:

```text
x-axis = actual value
y-axis = predicted value
```

The dashed diagonal line represents a perfect prediction.

R² is calculated using all valid validation values. At most 5,000 points are sampled only for drawing the scatter plot.

---

## 23. Current Validation R² Results

### 23.1 No-spike model

| Feature | Validation R² |
|---|---:|
| `voltage_base` | 0.990 |
| `steady_state_voltage` | 0.933 |
| `steady_state_voltage_stimend` | 0.857 |
| `decay_time_constant_after_stim` | 0.579 |
| `sag_amplitude` | Not enough valid data |

The first three voltage-related features are predicted well. The decay time constant is moderate. `sag_amplitude` cannot be evaluated reliably because too few valid validation values remain.

![No-spike training history](figures/regressor_history_no_spike.png)

![No-spike R2 summary](figures/regressor_r2_summary_no_spike.png)

![No-spike parity plot](figures/regressor_parity_no_spike.png)

### 23.2 Spiking model

| Feature | Validation R² |
|---|---:|
| `AP1_peak` | 0.930 |
| `voltage_base` | 0.899 |
| `AHP1_depth_from_peak` | 0.863 |
| `AHP_depth` | 0.801 |
| `mean_AP_amplitude` | 0.785 |
| `steady_state_voltage` | 0.667 |
| `Spikecount` | 0.633 |
| `steady_state_voltage_stimend` | 0.578 |
| `time_to_first_spike` | 0.569 |
| `inv_first_ISI` | 0.356 |
| `AP1_width` | 0.310 |
| `AHP_time_from_peak` | 0.191 |

The model is strong for spike peak, voltage base, and AHP depth features. Timing and width features remain more difficult.

![Spiking training history](figures/regressor_history_spiking.png)

![Spiking R2 summary](figures/regressor_r2_summary_spiking.png)

![Spiking parity plot](figures/regressor_parity_spiking.png)

### 23.3 Training-curve interpretation

In both regimes, training loss continues to decrease after validation loss begins to flatten.

This creates a train-validation gap:

- no-spike final training loss is lower than validation loss,
- spiking final training loss is also much lower than validation loss.

This suggests some overfitting. Early stopping restores the checkpoint with the best validation loss, but the plots still show the full recorded history until stopping.

---

# Part F — Checkpoint Output

## 24. Saved Regime Checkpoint

The default regressor command writes:

```text
regime_regressors.pt
```

The checkpoint contains:

- one model state per response class,
- class name,
- selected feature names,
- selected feature indices,
- training metrics,
- validation metrics,
- input and output dimensions,
- parameter and feature names,
- normalization statistics,
- drop settings.

### Current limitation

The checkpoint-writing section does not currently save `train_set.indices` or `val_set.indices`.

The sensitivity script first searches for saved validation indices. Because the current checkpoint normally does not contain them, the script uses its reproducible random fallback procedure instead of the exact training validation split.

---

# Part G — Sensitivity Analysis

## 25. Main Question

The sensitivity analysis asks:

> How strongly does the trained surrogate output change when one normalized input parameter changes slightly?

It calculates a local model-gradient score for every:

```text
response regime × target feature × input parameter
```

Amplitude is included as an input and receives a sensitivity score like the conductance and capacitance parameters.

---

## 26. Sensitivity Dataset Preparation

The script creates two versions of `MOODataset`.

### Normalized dataset

```python
normalize=True
apply_log_transform=True
```

This dataset supplies the exact normalized inputs expected by the trained model.

### Raw dataset

```python
normalize=False
apply_log_transform=False
```

This dataset supplies:

- raw `Spikecount` values for regime filtering,
- masks that identify valid target values.

Both datasets must have the same number and order of samples.

---

## 27. Choosing Samples for Each Regime

For each saved model, the script first tries to find saved validation indices.

It checks several possible checkpoint key names, such as:

- `validation_indices`
- `val_indices`
- `validation_indices_by_regime`
- `regime_val_indices`

If no saved indices exist, the current fallback is:

1. Randomly choose up to 300,000 candidate dataset indices with seed 42.
2. Read raw `Spikecount` values.
3. Keep only no-spike or spiking candidates for the current regime.
4. Stop after collecting up to 50,000 matching samples.

Therefore, the current sensitivity set is reproducible, but it is usually not the exact validation set used during training.

---

## 28. Automatic Device Selection

The sensitivity script selects the first available device in this order:

```text
CUDA → Apple MPS → CPU
```

The model and input batches are moved to this device.

Gradient sums are moved back to CPU and accumulated in `float64` for numerical stability.

---

## 29. Gradient Calculation

For each batch, the normalized input tensor is changed to:

```python
normalized_x.requires_grad_(True)
```

The model produces all selected target predictions.

For one target feature `f` and one input parameter `p`, the basic derivative is:

```text
∂f / ∂p
```

This value measures the local slope of the neural network.

- A large positive derivative means increasing the input locally increases the predicted feature.
- A large negative derivative means increasing the input locally decreases the predicted feature.
- A derivative near zero means the model output changes little at that sample.

The script sums the selected output over all valid rows in the batch and calls:

```python
torch.autograd.grad(...)
```

Because the network processes each row independently, the resulting input gradients give one gradient vector per sample.

---

## 30. Mean Absolute Gradient Score

Positive and negative gradients could cancel if they were averaged directly.

The script therefore uses absolute values:

```text
|∂f / ∂p|
```

For target feature `f`, input `p`, and `N` valid samples, the final score is:

```text
MAG(f, p) = (1 / N) × Σ |∂fᵢ / ∂pᵢ|
```

This is the Mean Absolute Gradient, or MAG.

The calculation steps are:

1. Select rows where the raw target mask is valid.
2. Calculate gradients of the selected model output with respect to all inputs.
3. Keep gradients only for valid rows.
4. Take the absolute value.
5. Sum the absolute gradients across the batch.
6. Add batch sums on the CPU.
7. Divide by the total valid-sample count.

The output matrix shape is:

```text
[number of selected target features, number of model inputs]
```

---

## 31. What Scale Does the Sensitivity Use?

The derivative is calculated between:

- normalized model inputs, and
- normalized model outputs.

Therefore, it is not a raw derivative such as “mV change per one Siemens change.”

It is closer to:

> change in normalized predicted feature per one normalized input unit.

This makes parameters with very different numerical scales easier to compare. However, the result describes the trained neural network, not a direct causal experiment on the biological simulator.

---

## 32. Valid Samples

Every sensitivity row stores `Valid_Samples`.

A target feature only uses samples where its raw mask is valid.

Different target features may therefore use different numbers of samples, even inside the same regime.

A score based on very few valid samples should be interpreted carefully.

---

## 33. Ranking

After all MAG scores are calculated, the script groups values by:

```text
Regime + Feature
```

It ranks input parameters in descending order:

```text
Rank 1 = largest MAG score
Rank 2 = second largest MAG score
...
```

Ties use the minimum rank.

The complete table is saved as:

```text
sensitivity_rankings.csv
```

Main columns are:

- `Regime`
- `Feature`
- `Parameter`
- `Sensitivity_Score`
- `Valid_Samples`
- `Rank`

---

## 34. Sensitivity Heatmaps

`plot_sensitivity.py` creates one heatmap per regime.

First, raw MAG scores are arranged as:

```text
rows = target features
columns = input parameters
```

Then each feature row is divided by its own largest score:

```text
relative_score(f, p) = MAG(f, p) / max over parameters MAG(f, p)
```

As a result:

```text
1.00 = most sensitive parameter for that feature
0.50 = half of that feature's maximum MAG
0.00 = little or no relative sensitivity
```

This row normalization is useful for finding the main drivers of one feature.

It does not preserve absolute differences between different feature rows. A `1.00` in one row may come from a much smaller raw MAG than a `0.50` in another row.

![No-spike sensitivity heatmap](figures/sensitivity_clean_heatmap_no_spike.png)

![Spiking sensitivity heatmap](figures/sensitivity_clean_heatmap_spiking.png)

---

## 35. Correct Interpretation of Sensitivity Results

The sensitivity ranking means:

> The trained surrogate is locally most responsive to this normalized input for this predicted feature, in the analyzed sample set.

It does not automatically prove:

- biological causation,
- global importance over the full parameter range,
- the direction of the effect,
- independence from correlated parameters,
- reliability for a target with low validation R².

The current MAG score also removes the sign. It measures strength, not whether the effect is positive or negative.

For strong scientific claims, sensitivity should be supported with:

- good validation R² for the target,
- enough valid samples,
- simulator-based parameter perturbation tests,
- repeated analysis across neurons or data splits,
- signed gradients or partial-dependence checks when direction matters.

---

# Part H — Running the Project

## 36. Required Python Packages

The uploaded code uses:

```text
numpy
pandas
torch
matplotlib
seaborn
```

Example installation:

```bash
pip install numpy pandas torch matplotlib seaborn
```

---

## 37. Suggested Folder Layout

```text
project/
├── data_analysis.py
├── data_pipline.py
├── multi_model.py
├── sensitivity_analysis.py
├── plot_sensitivity.py
├── regime_regressors.pt
├── sensitivity_rankings.csv
├── figures/
│   ├── regressor_history_no_spike.png
│   ├── regressor_history_spiking.png
│   ├── regressor_parity_no_spike.png
│   ├── regressor_parity_spiking.png
│   ├── regressor_r2_summary_no_spike.png
│   ├── regressor_r2_summary_spiking.png
│   ├── sensitivity_clean_heatmap_no_spike.png
│   └── sensitivity_clean_heatmap_spiking.png
└── test_data/
    ├── csv_dir/
    │   ├── ITL.../
    │   │   └── *_gen_*.csv
    │   └── ...
    └── outputs_dir/
        └── *protocol_stage*.json
```

---

## 38. Generate Data Statistics

The default paths are currently written inside `data_analysis.py`.

Run:

```bash
python data_analysis.py
```

This calculates new statistics and prints the saved tables.

To print existing CSV results without recomputing:

```bash
python data_analysis.py --show
```

Typical outputs include:

```text
parameter_ranges_summary4.csv
parameter_ranges_summary4_detailed_before.csv
parameter_ranges_summary4_detailed_after.csv
```

---

## 39. Train Both Regime Regressors

Example command:

```bash
python multi_model.py \
  ../test_data/csv_dir \
  ../test_data/outputs_dir \
  --regressors
```

This uses the main defaults:

```text
200 maximum epochs
0.001 learning rate
15 early-stopping patience
in-RAM training enabled
8192 effective in-RAM batch size
1,000,000 maximum samples per class
```

To change common settings:

```bash
python multi_model.py \
  ../test_data/csv_dir \
  ../test_data/outputs_dir \
  --regressors \
  --epochs 250 \
  --lr 0.0005 \
  --patience 20 \
  --ram-batch-size 4096
```

To use the older disk-backed training path:

```bash
python multi_model.py \
  ../test_data/csv_dir \
  ../test_data/outputs_dir \
  --regressors \
  --no-in-ram
```

The trained bundle is saved as:

```text
regime_regressors.pt
```

---

## 40. Run Sensitivity Analysis

The current paths and sample limits are set at the bottom of `sensitivity_analysis.py`.

Run:

```bash
python sensitivity_analysis.py
```

The default main block uses:

```text
checkpoint = regime_regressors.pt
maximum samples per regime = 50,000
fallback candidate pool = 300,000
seed = 42
```

Output:

```text
sensitivity_rankings.csv
```

---

## 41. Plot Sensitivity Heatmaps

Run:

```bash
python plot_sensitivity.py
```

Outputs:

```text
figures/sensitivity_clean_heatmap_no_spike.png
figures/sensitivity_clean_heatmap_spiking.png
```

---

# Part I — Important Code Review Notes

## 42. Points to Keep Consistent

### 42.1 Report cleaning and training cleaning differ

`data_analysis.py` and `data_pipline.py` use different local cleaning dictionaries. A shared configuration would make the reported data counts match the exact training data more closely.

### 42.2 Normalization cleaning and item cleaning differ

Normalization statistics use cell-wise bound invalidation. Final dataset samples use row-wise invalidation. Both stages should use the same rule.

### 42.3 The residual regressor uses one width

Although `hidden_dims` is a tuple, the regressor only uses `hidden_dims[0]`. Changing 128 or 64 currently has no effect on `FeaturePredictor`.

### 42.4 Validation indices are not saved

The checkpoint does not save the exact validation indices. Sensitivity therefore normally uses a random regime-matched subset.

### 42.5 Heatmap scores are relative within each row

The displayed heatmap values are not raw MAG scores. They are divided by each feature's own maximum.

### 42.6 Sensitivity reliability depends on model quality

Sensitivity for low-R² targets such as `AHP_time_from_peak`, `AP1_width`, and `inv_first_ISI` should be treated as exploratory.

### 42.7 `sag_amplitude` may be over-filtered

The strict 7.0 mV upper bound combined with full-row invalidation can remove many no-spike rows. This should be reviewed against the raw distribution and the biological definition used by eFEL.

### 42.8 Feature locations are collapsed

The parser reads `protocol.location.feature`, but the final target vector keeps only the canonical feature name. If the same protocol contains the same feature at more than one location, a later column can overwrite an earlier one in the feature block.

---

# Part J — Summary

The current system is a large-scale, regime-specific neural surrogate pipeline.

It:

- reads multi-neuron MOO generation data,
- expands protocols and adds stimulation amplitude,
- creates masks for missing features,
- applies physiological filtering,
- transforms long-tailed targets,
- normalizes inputs and targets,
- separates no-spike and spiking samples,
- trains two residual MLP regressors,
- evaluates predictions in raw physiological units,
- and ranks input importance using Mean Absolute Gradients.

The strongest current prediction results are for voltage and spike-amplitude-related features. Several timing features remain difficult. The sensitivity analysis is useful for exploring parameter-feature relationships, but its conclusions should be considered together with validation R², valid-sample counts, cleaning behavior, and direct simulator checks.
