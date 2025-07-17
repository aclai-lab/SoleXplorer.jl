using Test
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = load_arff_dataset("NATOPS")

modelts = symbolic_analysis(
    Xts, yts;
    model=ModalDecisionTree(),
    resample=(type=Holdout(shuffle=true), train_ratio=0.5, rng=Xoshiro(1)),
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    modalreduce=mean,
    features=(maximum, minimum),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)

# I'm easy like sunday morning
modelc = symbolic_analysis(Xc, yc)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                               usage example #1                               #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)    
)
solemc = train_test(dsc)
modelc = symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsr = setup_dataset(
    Xr, yr;
    model=DecisionTreeRegressor(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)    
)
solemr = train_test(dsr)
modelr = symbolic_analysis(
    dsr, solemr;
    measures=(rms, l1, l2, mae, mav)
)
@test modelr isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                               usage example #2                               #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2),
    # extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelr = symbolic_analysis(
    Xr, yr;
    model=DecisionTreeRegressor(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2),
    measures=(rms, l1, l2, mae, mav)
)
@test modelr isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                         resamples in numeric datasets                        #
# ---------------------------------------------------------------------------- #
modelc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=(type=Holdout(shuffle=true), train_ratio=0.75, rng=Xoshiro(1)),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelc isa SX.ModelSet

modelr = symbolic_analysis(
    Xr, yr;
    model=RandomForestRegressor(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    measures=(rms, l1, l2, mae, mav)      
)
@test modelr isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=AdaBoostStumpClassifier(),
    resample=(type=StratifiedCV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#              resamples in propositional translated time series               #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=DecisionTreeClassifier(),
    resample=(type=Holdout(shuffle=true), train_ratio=0.5, rng=Xoshiro(1)),
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    modalreduce=mean,
    features=(maximum, minimum),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=RandomForestClassifier(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    modalreduce=mean,
    features=(maximum, minimum),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=AdaBoostStumpClassifier(),
    resample=(type=StratifiedCV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    modalreduce=mean,
    features=(maximum, minimum),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet

# TODO known bug, see TODO.md 
# modelts = symbolic_analysis(
#     Xts, yts;
#     model=XGBoostClassifier(),
#     resample=(type=TimeSeriesCV(nfolds=5), rng=Xoshiro(1)),
#     win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
#     modalreduce=mean,
#     features=(maximum, minimum),
#     measures=(accuracy, log_loss, confusion_matrix, kappa)
# )
# @test modelts isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                        resample in modal time series                         #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=ModalDecisionTree(),
    resample=(type=CV(;nfolds=4), rng=Xoshiro(1)),
    measures=(accuracy,)
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=ModalRandomForest(),
    resample=(type=Holdout(shuffle=true), train_ratio=0.75, rng=Xoshiro(1)),
    features=(minimum, maximum),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
@test modelts isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#             xgboost makewatchlist for early stopping technique               #
# ---------------------------------------------------------------------------- #
range = SX.range(:num_round; lower=10, unit=10, upper=100)
modelr = symbolic_analysis(
    Xr, yr;
    model=XGBoostRegressor(
        early_stopping_rounds=20,
    ),
    resample=(type=CV(nfolds=5, shuffle=true), valid_ratio=0.2, rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2),
    measures=(rms, l1, l2, mae, mav) 
)
@test modelr isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                              catch9 and catch22                              #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=DecisionTreeClassifier(),
    resample=(;rng=Xoshiro(1)),
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
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=DecisionTreeClassifier(),
    resample=(;rng=Xoshiro(1)),
    features=(base_set...,),
    measures=(accuracy,)
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=DecisionTreeClassifier(),
    resample=(;rng=Xoshiro(1)),
    features=(catch9...,),
    measures=(accuracy,)
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=DecisionTreeClassifier(),
    resample=(;rng=Xoshiro(1)),
    features=(catch22_set...,),
    measures=(accuracy,)
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=DecisionTreeClassifier(),
    resample=(;rng=Xoshiro(1)),
    features=(complete_set...,),
    measures=(accuracy,)
)
@test modelts isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                                    tuning                                    #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelts = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
@test modelts isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelts = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    tuning=(;tuning=RandomSearch(), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
@test modelts isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelts = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    tuning=(;tuning=RandomSearch(), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
@test modelts isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelts = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    tuning=(;tuning=ParticleSwarm(), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
@test modelts isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelts = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    tuning=(;tuning=AdaptiveParticleSwarm(), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
@test modelts isa SX.ModelSet

# selector = FeatureSelector()
# range = MLJ.range(selector, :features, values = ((:sepal_length,), (:sepal_length, :sepal_width)))
# iterator(r2)
