using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                           propositional time series                          #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:xgboost),
    preprocess=(;rng=Xoshiro(1)),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
@test modelts isa SoleXplorer.Modelset
@test modelts.measures.measures isa Vector

modelts = symbolic_analysis(
    Xts, yts;
    model=(type=:decisiontree, params=(;max_depth=5, modalreduce=maximum)),
)
@test modelts isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                               modal time series                              #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:modaldecisiontree)
)
@test modelts isa SoleXplorer.Modelset

modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:modaldecisiontree),
    preprocess=(;rng=Xoshiro(1)),
    features=(minimum, maximum),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#             xgboost makewatchlist for early stopping technique               #
# ---------------------------------------------------------------------------- #
early_stop  = symbolic_analysis(
    Xc, yc; 
    model=(
        type=:xgboost_classifier,
        params=(
            num_round=100,
            max_depth=6,
            eta=0.1, 
            objective="multi:softprob",
            # early_stopping parameters
            early_stopping_rounds=10,
            watchlist=makewatchlist
        )
    ),
    # with early stopping a validation set is required
    preprocess=(valid_ratio = 0.3, rng=Xoshiro(1))
)
@test early_stop isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                              catch9 and catch22                              #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(1)),
    features=(
        mode_5,
        mode_10,
        embedding_dist,
        acf_timescale,
        acf_first_min,
        ami2,
        trev,
        outlier_timing_pos,
        outlier_timing_neg,
        whiten_timescale,
        forecast_error,
        ami_timescale,
        high_fluctuation,
        stretch_decreasing,
        stretch_high,
        entropy_pairs,
        rs_range,
        dfa,
        low_freq_power,
        centroid_freq,
        transition_variance,
        periodicity
    ),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset

modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(1)),
    features=(base_set...,),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset

modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(1)),
    features=(catch9...,),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset

modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(1)),
    features=(catch22_set...,),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset

modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(1)),
    features=(complete_set...,),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                                    tuning                                    #
# ---------------------------------------------------------------------------- #
modelc = symbolic_analysis(
    Xc, yc;
    tuning=true
)
@test modelts isa SoleXplorer.Modelset

modelc = symbolic_analysis(
    Xc, yc;
    tuning=(
        method=(type=grid, params=(;resolution=25)), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    )
)
@test modelts isa SoleXplorer.Modelset

modelc = symbolic_analysis(
    Xc, yc;
    tuning=(
        method=(;type=randomsearch), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    )
)
@test modelts isa SoleXplorer.Modelset

modelc = symbolic_analysis(
    Xc, yc;
    tuning=(
        method=(type=latinhypercube, params=(;popsize=80)), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    )
)
@test modelts isa SoleXplorer.Modelset

modelc = symbolic_analysis(
    Xc, yc;
    tuning=(
        method=(type=particleswarm, params=(;n_particles=5)), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    )
)
@test modelts isa SoleXplorer.Modelset

modelc = symbolic_analysis(
    Xc, yc;
    tuning=(
        method=(;type=adaptiveparticleswarm), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    )
)
@test modelts isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                              rules extraction                                #
# ---------------------------------------------------------------------------- #
modelc = symbolic_analysis(
    Xc, yc;
    model=(;type=:randomforest),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:intrees)
)
@test modelc isa SoleXplorer.Modelset

modelc = symbolic_analysis(
    Xc, yc;
    model=(;type=:randomforest),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:refne)
)
@test modelc isa SoleXplorer.Modelset

modelc = symbolic_analysis(
    Xc, yc;
    model=(;type=:randomforest),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:trepan)
)
@test modelc isa SoleXplorer.Modelset

# TODO: broken
# modelc = symbolic_analysis(
#     Xc, yc;
#     model=(;type=:randomforest),
#     preprocess=(;rng=Xoshiro(1)),
#     extract_rules=(;type=:batrees)
# )
# @test modelc isa SoleXplorer.Modelset

modelc = symbolic_analysis(
    Xc, yc;
    model=(;type=:randomforest),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:rulecosi)
)
@test modelc isa SoleXplorer.Modelset

modelc = symbolic_analysis(
    Xc, yc;
    model=(;type=:randomforest),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:lumen)
)
@test modelc isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                   check modal features correctly in model                    #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:modaldecisiontree),
    features=(forecast_error, dfa)
)
# @test modelts.mach.model.conditions == [forecast_error, dfa]

# ---------------------------------------------------------------------------- #
#                       check statisticalmeasure Sum()                         #
# ---------------------------------------------------------------------------- #
modelr = symbolic_analysis(
    Xr, yr;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(1)),
    measures=(rmse, l1, l1_sum,)
)
@test modelts isa SoleXplorer.Modelset
@test modelts.measures.measures isa Vector