# References
# The Catch22 features are based on the CAnonical Time-series CHaracteristics from:
# - Repository: https://github.com/DynamicsAndNeuralSystems/catch22
# - Article:    https://doi.org/10.1007/s10618-019-00647-x
# - Author: Carl H. Lubba et al

# ---------------------------------------------------------------------------- #
#                        catch22 pretty named functions                        #
# ---------------------------------------------------------------------------- #
mode_5(x)              = Catch22.DN_HistogramMode_5((x));                              
    @doc (@doc Catch22.DN_HistogramMode_5) mode_5
mode_10(x)             = Catch22.DN_HistogramMode_10((x));                            
    @doc (@doc Catch22.DN_HistogramMode_10) mode_10
embedding_dist(x)      = Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff((x));    
    @doc (@doc Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff) embedding_dist
acf_timescale(x)       = Catch22.CO_f1ecac((x));                                
    @doc (@doc Catch22.CO_f1ecac) acf_timescale
acf_first_min(x)       = Catch22.CO_FirstMin_ac((x));                           
    @doc (@doc Catch22.CO_FirstMin_ac) acf_first_min
ami2(x)                = Catch22.CO_HistogramAMI_even_2_5((x));                          
    @doc (@doc Catch22.CO_HistogramAMI_even_2_5) ami2
trev(x)                = Catch22.CO_trev_1_num((x));                                     
    @doc (@doc Catch22.CO_trev_1_num) trev
outlier_timing_pos(x)  = Catch22.DN_OutlierInclude_p_001_mdrmd((x));       
    @doc (@doc Catch22.DN_OutlierInclude_p_001_mdrmd) outlier_timing_pos
outlier_timing_neg(x)  = Catch22.DN_OutlierInclude_n_001_mdrmd((x));       
    @doc (@doc Catch22.DN_OutlierInclude_n_001_mdrmd) outlier_timing_neg
whiten_timescale(x)    = Catch22.FC_LocalSimple_mean1_tauresrat((x));        
    @doc (@doc Catch22.FC_LocalSimple_mean1_tauresrat) whiten_timescale
forecast_error(x)      = Catch22.FC_LocalSimple_mean3_stderr((x));             
    @doc (@doc Catch22.FC_LocalSimple_mean3_stderr) forecast_error
ami_timescale(x)       = Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi((x));  
    @doc (@doc Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi) ami_timescale
high_fluctuation(x)    = Catch22.MD_hrv_classic_pnn40((x));                  
    @doc (@doc Catch22.MD_hrv_classic_pnn40) high_fluctuation
stretch_decreasing(x)  = Catch22.SB_BinaryStats_diff_longstretch0((x));    
    @doc (@doc Catch22.SB_BinaryStats_diff_longstretch0) stretch_decreasing
stretch_high(x)        = Catch22.SB_BinaryStats_mean_longstretch1((x));          
    @doc (@doc Catch22.SB_BinaryStats_mean_longstretch1) stretch_high
entropy_pairs(x)       = Catch22.SB_MotifThree_quantile_hh((x));                
    @doc (@doc Catch22.SB_MotifThree_quantile_hh) entropy_pairs
rs_range(x)            = Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1((x));   
    @doc (@doc Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1) rs_range
dfa(x)                 = Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1((x));             
    @doc (@doc Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1) dfa
low_freq_power(x)      = Catch22.SP_Summaries_welch_rect_area_5_1((x));        
    @doc (@doc Catch22.SP_Summaries_welch_rect_area_5_1) low_freq_power
centroid_freq(x)       = Catch22.SP_Summaries_welch_rect_centroid((x));         
    @doc (@doc Catch22.SP_Summaries_welch_rect_centroid) centroid_freq
transition_variance(x) = Catch22.SB_TransitionMatrix_3ac_sumdiagcov((x)); 
    @doc (@doc Catch22.SB_TransitionMatrix_3ac_sumdiagcov) transition_variance
periodicity(x)         = Catch22.PD_PeriodicityWang_th0_01((x));                  
    @doc (@doc Catch22.PD_PeriodicityWang_th0_01) periodicity

# ---------------------------------------------------------------------------- #
#                                  featuresets                                 #
# ---------------------------------------------------------------------------- #
"""
    base_set

A minimal feature set containing only basic statistical measures for time series analysis.

# Features
- `maximum`: Maximum value in the time series
- `minimum`: Minimum value in the time series  
- `mean`   : Arithmetic mean of the time series
- `std`    : Standard deviation of the time series

"""
base_set     = (maximum, minimum, MLJ.mean, MLJ.std)

"""
    catch9

A curated subset of 9 features combining basic statistics with Symbolic Catch22 measures,

# Features
- Basic statistics: `maximum`, `minimum`, `mean`, `median`, `std`
- Symbolic Catch22 features:
  - `stretch_high`       : Measures persistence of high values
  - `stretch_decreasing` : Captures decreasing trend patterns
  - `entropy_pairs`      : Quantifies local pattern complexity
  - `transition_variance`: Measures state transition variability

# References
The Catch22 features are based on the CAnonical Time-series CHaracteristics from:
- Repository: https://github.com/DynamicsAndNeuralSystems/catch22
- Article:    https://doi.org/10.1007/s10618-019-00647-x
- Author: Carl H. Lubba et al

"""
catch9       = (maximum, minimum, MLJ.mean, MLJ.median, MLJ.std,
                stretch_high, stretch_decreasing, entropy_pairs, transition_variance)

                """
    catch22_set

The complete Catch22 feature set.
Each feature captures different aspects of time series dynamics including
correlation structure, distribution properties and temporal patterns.

# Feature Categories
- **Distribution shape**: 
  `mode_5`, `mode_10`
- **Extreme event timing**:
  `outlier_timing_pos`, `outlier_timing_neg`
- **Linear autocorrelation**:
  `acf_timescale`, `acf_first_min`, `low_freq_power`, `centroid_freq`
- **Simple forecasting**:
  `forecast_error`
- **Incremental differences**:
  `whiten_timescale`, `high_fluctuation`
- **Symbolic**
  `stretch_high`, `stretch_decreasing`, `entropy_pairs`, `transition_variance`
- **Nonlinear autocorrelation**:
  `ami2`, `trev`
- **Linear autocorrelation structure**:
  `ami_timescale`, `periodicity`
- **Self-affine scaling**:
  `rs_range`, `dfa`
- **Other**:
  `embedding_dist`

# References
The Catch22 features are based on the CAnonical Time-series CHaracteristics from:
- Repository: https://github.com/DynamicsAndNeuralSystems/catch22
- Article:    https://doi.org/10.1007/s10618-019-00647-x
- Author: Carl H. Lubba et al

"""
catch22_set  = (mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2,
                trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale, 
                forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
                stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, 
                centroid_freq, transition_variance, periodicity)

"""
    complete_set

The most comprehensive feature set combining basic statistical measures, covariance
analysis, and the full Catch22 suite.

# Features
- **Basic statistics**:
  `maximum`, `minimum`, `mean`, `median`, `std`, `cov`
- **Distribution shape**: 
  `mode_5`, `mode_10`
- **Extreme event timing**:
  `outlier_timing_pos`, `outlier_timing_neg`
- **Linear autocorrelation**:
  `acf_timescale`, `acf_first_min`, `low_freq_power`, `centroid_freq`
- **Simple forecasting**:
  `forecast_error`
- **Incremental differences**:
  `whiten_timescale`, `high_fluctuation`
- **Symbolic**
  `stretch_high`, `stretch_decreasing`, `entropy_pairs`, `transition_variance`
- **Nonlinear autocorrelation**:
  `ami2`, `trev`
- **Linear autocorrelation structure**:
  `ami_timescale`, `periodicity`
- **Self-affine scaling**:
  `rs_range`, `dfa`
- **Other**:
  `embedding_dist`

# References
The Catch22 features are based on the CAnonical Time-series CHaracteristics from:
- Repository: https://github.com/DynamicsAndNeuralSystems/catch22
- Article:    https://doi.org/10.1007/s10618-019-00647-x
- Author: Carl H. Lubba et al
"""
complete_set = (maximum, minimum, MLJ.mean, MLJ.median, MLJ.std,
                MLJ.StatsBase.cov, mode_5, mode_10, embedding_dist, acf_timescale,
                acf_first_min, ami2, trev, outlier_timing_pos, outlier_timing_neg,
                whiten_timescale, forecast_error, ami_timescale, high_fluctuation,
                stretch_decreasing, stretch_high, entropy_pairs, rs_range, dfa,
                low_freq_power, centroid_freq, transition_variance, periodicity)