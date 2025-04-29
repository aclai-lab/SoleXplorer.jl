using SoleXplorer
using Test
using StatsBase
using Catch22

@testset "features_set.jl" begin
    # test time series
    ts = randn(100)
    
    @testset "Individual wrapper functions" begin
        @test mode_5(ts)              == Catch22.DN_HistogramMode_5(ts)
        @test embedding_dist(ts)      == Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff(ts)
        @test acf_timescale(ts)       == Catch22.CO_f1ecac(ts)
        @test acf_first_min(ts)       == Catch22.CO_FirstMin_ac(ts)
        @test ami2(ts)                == Catch22.CO_HistogramAMI_even_2_5(ts)
        @test trev(ts)                == Catch22.CO_trev_1_num(ts)
        @test outlier_timing_pos(ts)  == Catch22.DN_OutlierInclude_p_001_mdrmd(ts)
        @test outlier_timing_neg(ts)  == Catch22.DN_OutlierInclude_n_001_mdrmd(ts)
        @test whiten_timescale(ts)    == Catch22.FC_LocalSimple_mean1_tauresrat(ts)
        @test forecast_error(ts)      == Catch22.FC_LocalSimple_mean3_stderr(ts)
        @test ami_timescale(ts)       == Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(ts)
        @test high_fluctuation(ts)    == Catch22.MD_hrv_classic_pnn40(ts)
        @test stretch_decreasing(ts)  == Catch22.SB_BinaryStats_diff_longstretch0(ts)
        @test stretch_high(ts)        == Catch22.SB_BinaryStats_mean_longstretch1(ts)
        @test entropy_pairs(ts)       == Catch22.SB_MotifThree_quantile_hh(ts)
        @test rs_range(ts)            == Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(ts)
        @test dfa(ts)                 == Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(ts)
        @test low_freq_power(ts)      == Catch22.SP_Summaries_welch_rect_area_5_1(ts)
        @test centroid_freq(ts)       == Catch22.SP_Summaries_welch_rect_centroid(ts)
        @test transition_variance(ts) == Catch22.SB_TransitionMatrix_3ac_sumdiagcov(ts)
        @test periodicity(ts)         == Catch22.PD_PeriodicityWang_th0_01(ts)
    end
    
    @testset "Feature sets" begin
        # extract features
        base_features     = [f(ts) for f in base_set]
        catch9_features   = [f(ts) for f in catch9]
        catch22_features  = [f(ts) for f in catch22_set]
        complete_features = [f(ts) for f in complete_set]

        @test length(base_features)     == 4
        @test length(catch9_features)   == 9
        @test length(catch22_features)  == 22
        @test length(complete_features) == 28

        @test base_features     == [maximum(ts), minimum(ts), StatsBase.mean(ts), StatsBase.std(ts)]

        @test catch9_features   == [maximum(ts), minimum(ts), StatsBase.mean(ts), StatsBase.median(ts),
                                    StatsBase.std(ts), stretch_high(ts), stretch_decreasing(ts),
                                    entropy_pairs(ts), transition_variance(ts)]

        @test catch22_features  == [mode_5(ts), mode_10(ts), embedding_dist(ts), acf_timescale(ts),
                                    acf_first_min(ts), ami2(ts), trev(ts), outlier_timing_pos(ts),
                                    outlier_timing_neg(ts), whiten_timescale(ts), forecast_error(ts),
                                    ami_timescale(ts), high_fluctuation(ts), stretch_decreasing(ts),
                                    stretch_high(ts), entropy_pairs(ts), rs_range(ts), dfa(ts),
                                    low_freq_power(ts), centroid_freq(ts), transition_variance(ts),
                                    periodicity(ts)]

        @test complete_features == [maximum(ts), minimum(ts), StatsBase.mean(ts), StatsBase.median(ts),
                                    StatsBase.std(ts), StatsBase.cov(ts), mode_5(ts), mode_10(ts), 
                                    embedding_dist(ts), acf_timescale(ts), acf_first_min(ts), ami2(ts),
                                    trev(ts), outlier_timing_pos(ts), outlier_timing_neg(ts),
                                    whiten_timescale(ts), forecast_error(ts), ami_timescale(ts),
                                    high_fluctuation(ts), stretch_decreasing(ts), stretch_high(ts),
                                    entropy_pairs(ts), rs_range(ts), dfa(ts), low_freq_power(ts),
                                    centroid_freq(ts), transition_variance(ts), periodicity(ts)]
    end
end