# ---------------------------------------------------------------------------- #
#                        catch22 pretty named functions                        #
# ---------------------------------------------------------------------------- #

mode_5(x) = Catch22.DN_HistogramMode_5((x));                              @doc (@doc Catch22.DN_HistogramMode_5) mode_5
mode_10(x) = Catch22.DN_HistogramMode_10((x));                            @doc (@doc Catch22.DN_HistogramMode_10) mode_10
embedding_dist(x) = Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff((x));    @doc (@doc Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff) embedding_dist
acf_timescale(x) = Catch22.CO_f1ecac((x));                                @doc (@doc Catch22.CO_f1ecac) acf_timescale
acf_first_min(x) = Catch22.CO_FirstMin_ac((x));                           @doc (@doc Catch22.CO_FirstMin_ac) acf_first_min
ami2(x) = Catch22.CO_HistogramAMI_even_2_5((x));                          @doc (@doc Catch22.CO_HistogramAMI_even_2_5) ami2
trev(x) = Catch22.CO_trev_1_num((x));                                     @doc (@doc Catch22.CO_trev_1_num) trev
outlier_timing_pos(x) = Catch22.DN_OutlierInclude_p_001_mdrmd((x));       @doc (@doc Catch22.DN_OutlierInclude_p_001_mdrmd) outlier_timing_pos
outlier_timing_neg(x) = Catch22.DN_OutlierInclude_n_001_mdrmd((x));       @doc (@doc Catch22.DN_OutlierInclude_n_001_mdrmd) outlier_timing_neg
whiten_timescale(x) = Catch22.FC_LocalSimple_mean1_tauresrat((x));        @doc (@doc Catch22.FC_LocalSimple_mean1_tauresrat) whiten_timescale
forecast_error(x) = Catch22.FC_LocalSimple_mean3_stderr((x));             @doc (@doc Catch22.FC_LocalSimple_mean3_stderr) forecast_error
ami_timescale(x) = Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi((x));  @doc (@doc Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi) ami_timescale
high_fluctuation(x) = Catch22.MD_hrv_classic_pnn40((x));                  @doc (@doc Catch22.MD_hrv_classic_pnn40) high_fluctuation
stretch_decreasing(x) = Catch22.SB_BinaryStats_diff_longstretch0((x));    @doc (@doc Catch22.SB_BinaryStats_diff_longstretch0) stretch_decreasing
stretch_high(x) = Catch22.SB_BinaryStats_mean_longstretch1((x));          @doc (@doc Catch22.SB_BinaryStats_mean_longstretch1) stretch_high
entropy_pairs(x) = Catch22.SB_MotifThree_quantile_hh((x));                @doc (@doc Catch22.SB_MotifThree_quantile_hh) entropy_pairs
rs_range(x) = Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1((x));   @doc (@doc Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1) rs_range
dfa(x) = Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1((x));             @doc (@doc Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1) dfa
low_freq_power(x) = Catch22.SP_Summaries_welch_rect_area_5_1((x));        @doc (@doc Catch22.SP_Summaries_welch_rect_area_5_1) low_freq_power
centroid_freq(x) = Catch22.SP_Summaries_welch_rect_centroid((x));         @doc (@doc Catch22.SP_Summaries_welch_rect_centroid) centroid_freq
transition_variance(x) = Catch22.SB_TransitionMatrix_3ac_sumdiagcov((x)); @doc (@doc Catch22.SB_TransitionMatrix_3ac_sumdiagcov) transition_variance
periodicity(x) = Catch22.PD_PeriodicityWang_th0_01((x));                  @doc (@doc Catch22.PD_PeriodicityWang_th0_01) periodicity

# ---------------------------------------------------------------------------- #
#                                     catch9                                   #
# ---------------------------------------------------------------------------- #
base_set = [maximum, minimum, mean, std]
catch9 = [maximum, minimum, mean, median, std, stretch_high, stretch_decreasing, entropy_pairs, transition_variance]
catch22_set = [mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos,
    outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
    stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity]
complete_set = [maximum, minimum, mean, median, std, StatsBase.cov,
    mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos,
    outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
    stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity]