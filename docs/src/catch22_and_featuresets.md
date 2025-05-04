```@meta
CurrentModule = SoleXplorer
```
```@contents
Pages = ["catch22_and_featuresets.md"]
```

# [Catch22 Feature Extraction Functions](@id catch22-feature-extraction)

This module provides user-friendly wrapper functions for the [Catch22](https://github.com/chlubba/catch22) time series feature extraction library. Catch22 includes 22 time series features that have been selected for their high performance in time series classification tasks.

## [Overview](@id catch22-overview)

These wrapper functions offer intuitive, shorter names for the Catch22 feature extraction functions while preserving their original functionality. Each wrapper inherits the full documentation from its corresponding Catch22 function.

## [Available Feature Functions](@id catch22-functions)

| Wrapper Function | Description |
|------------------|-------------|
| `mode_5(x)` | Mode of z-score distribution (5-bin histogram) |
| `mode_10(x)` | Mode of z-score distribution (10-bin histogram) |
| `embedding_dist(x)` | Mean distance between successive embedding coordinates |
| `acf_timescale(x)` | First 1/e crossing of autocorrelation function |
| `acf_first_min(x)` | First minimum of autocorrelation function |
| `ami2(x)` | Automutual information with 2-bin histogram |
| `trev(x)` | Time-reversibility statistic |
| `outlier_timing_pos(x)` | Timing of positive outliers |
| `outlier_timing_neg(x)` | Timing of negative outliers |
| `whiten_timescale(x)` | AR model estimation timescale |
| `forecast_error(x)` | Forecast error of AR model |
| `ami_timescale(x)` | First minimum of automutual information function |
| `high_fluctuation(x)` | Proportion of high fluctuations |
| `stretch_decreasing(x)` | Longest stretch of decreasing values |
| `stretch_high(x)` | Longest stretch of values above mean |
| `entropy_pairs(x)` | Entropy of successive motif patterns |
| `rs_range(x)` | Range of rescaled-range analysis |
| `dfa(x)` | Detrended fluctuation analysis exponent |
| `low_freq_power(x)` | Low-frequency power in Fourier spectrum |
| `centroid_freq(x)` | Centroid frequency of the power spectrum |
| `transition_variance(x)` | Variance of transition matrix diagonal |
| `periodicity(x)` | Periodicity measure using Wang's method |

## [Predefined Feature Sets](@id catch22-feature-sets)

The module provides several predefined sets of features for convenience:

### [`base_set`](@id base-feature-set)
Basic statistical measures:
```julia
(maximum, minimum, MLJ.mean, MLJ.std)
```

### [`catch9`](@id catch9-feature-set)
A subset of 9 features (includes basic statistics plus selected Catch22 features):
```julia
(maximum, minimum, MLJ.mean, MLJ.median, MLJ.std,
 stretch_high, stretch_decreasing, entropy_pairs, transition_variance)
```

### [`catch22_set`](@id catch22-feature-set)
All 22 Catch22 features:
```julia
(mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2,
 trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale, 
 forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
 stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, 
 centroid_freq, transition_variance, periodicity)
```

### [`complete_set`](@id complete-feature-set)
Basic statistics plus all Catch22 features:
```julia
(maximum, minimum, MLJ.mean, MLJ.median, MLJ.std,
 MLJ.StatsBase.cov, mode_5, mode_10, embedding_dist, acf_timescale,
 acf_first_min, ami2, trev, outlier_timing_pos, outlier_timing_neg,
 whiten_timescale, forecast_error, ami_timescale, high_fluctuation,
 stretch_decreasing, stretch_high, entropy_pairs, rs_range, dfa,
 low_freq_power, centroid_freq, transition_variance, periodicity)
```

## [Usage Example](@id catch22-usage)

```julia
using SoleXplorer

# Calculate a single feature
ts = rand(100)
periodicity_value = periodicity(ts)

# Calculate all Catch22 features
results = [f(ts) for f in catch22_set]

# Use complete feature set for analysis
all_features = [f(ts) for f in complete_set]
```

```@docs
mode_5
mode_10
embedding_dist
acf_timescale
acf_first_min
ami2
trev
outlier_timing_pos
outlier_timing_neg
whiten_timescale
forecast_error
ami_timescale
high_fluctuation
stretch_decreasing
stretch_high
entropy_pairs
rs_range
dfa
low_freq_power
centroid_freq
transition_variance
periodicity
```