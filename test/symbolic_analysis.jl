using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = NatopsLoader()
Xts, yts = SX.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                        I'm easy like sunday morning                          #
# ---------------------------------------------------------------------------- #
modelc = symbolic_analysis(Xc, yc)
@test modelc isa SX.ModelSet

modelr = symbolic_analysis(Xr, yr)
@test modelc isa SX.ModelSet

modelts = symbolic_analysis(Xts, yts)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                               usage example #1                               #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsc = setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(;range, resolution=10, resampling=CV(nfolds=3), measure=SX.accuracy, repeats=2)    
)
solemc = train_test(dsc)
modelc = symbolic_analysis(
    dsc, solemc;
    extractor=SX.InTreesRuleExtractor(),
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsr = setup_dataset(
    Xr, yr;
    model=SX.DecisionTreeRegressor(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=20, resampling=CV(nfolds=3), range=range, repeats=2)    
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
    model=SX.DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2),
    extractor=SX.InTreesRuleExtractor(),
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelr = symbolic_analysis(
    Xr, yr;
    model=SX.DecisionTreeRegressor(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(; range, resolution=10, resampling=CV(nfolds=3), measure=rms, repeats=2),
    measures=(rms, l1, l2, mae, mav)
)
@test modelr isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                    time series dataset with no windowing                     #
# ---------------------------------------------------------------------------- #
dsts = setup_dataset(
    Xts, yts;
    model=SX.ModalDecisionTree(),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    features=()  
)
# ---------------------------------------------------------------------------- #
#                        resamplings in numeric datasets                       #
# ---------------------------------------------------------------------------- #
modelc = symbolic_analysis(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    resampling=Holdout(fraction_train=0.75, shuffle=true),
    seed=1,
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelc isa SX.ModelSet

modelr = symbolic_analysis(
    Xr, yr;
    model=SX.RandomForestRegressor(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    measures=(rms, l1, l2, mae, mav)      
)
@test modelr isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    seed=1,
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#             resamplings in propositional translated time series              #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    resampling=Holdout(fraction_train=0.5, shuffle=true),
    seed=1,
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    modalreduce=mean,
    features=(maximum, minimum),
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.RandomForestClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    modalreduce=mean,
    features=(maximum, minimum),
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.AdaBoostStumpClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    seed=1,
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    modalreduce=mean,
    features=(maximum, minimum),
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet

# TODO known bug, see TODO.md 
# modelts = symbolic_analysis(
#     Xts, yts;
#     model=SX.XGBoostClassifier(),
#     resampling=(type=TimeSeriesCV(nfolds=5), seed=1),
#     win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
#     modalreduce=mean,
#     features=(maximum, minimum),
#     measures=(SX.accuracy, log_loss, confusion_matrix, kappa)
# )
# @test modelts isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                       resampling in modal time series                        #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=SX.ModalDecisionTree(),
    resampling=CV(;nfolds=4),
    seed=1,
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.ModalRandomForest(),
    resampling=Holdout(fraction_train=0.75, shuffle=true),
    seed=1,
    features=(minimum, maximum),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelts isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#             xgboost makewatchlist for early stopping technique               #
# ---------------------------------------------------------------------------- #
modelc = symbolic_analysis(
    Xc, yc;
    model=SX.XGBoostClassifier(early_stopping_rounds=20),
    resampling=CV(nfolds=5, shuffle=true),
    valid_ratio=0.2,
    seed=1,
    measures=(confusion_matrix,) 
)
@test modelc isa SX.ModelSet

range = SX.range(:num_round; lower=10, unit=10, upper=100)
modelr = symbolic_analysis(
    Xr, yr;
    model=SX.XGBoostRegressor(early_stopping_rounds=20),
    resampling=CV(nfolds=5, shuffle=true),
    valid_ratio=0.2,
    seed=1,
    tuning=GridTuning(; range, resolution=10, resampling=CV(nfolds=3), measure=rms, repeats=2),
    measures=(rms, l1, l2, mae, mav) 
)
@test modelr isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                              catch9 and catch22                              #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
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
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    features=(base_set...,),
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    features=(catch9...,),
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    features=(catch22_set...,),
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    features=(complete_set...,),
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                                  balancing                                   #
# ---------------------------------------------------------------------------- #
modelc = symbolic_analysis(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=BorderlineSMOTE1(m=6, k=4),
        undersampler=ClusterUndersampler()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.RandomForestClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=ENNUndersampler(k=7),
        undersampler=ROSE()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=RandomOversampler(),
        undersampler=RandomUndersampler()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.ModalDecisionTree(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=RandomWalkOversampler(),
        undersampler=SMOTE()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.ModalRandomForest(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTE(),
        undersampler=RandomUndersampler()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=ModalAdaBoost(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTENC(),
        undersampler=TomekUndersampler()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.XGBoostClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTENC(),
        undersampler=TomekUndersampler()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

@test_throws ArgumentError symbolic_analysis(
    Xr, yr;
    model=SX.RandomForestRegressor(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=ENNUndersampler(k=7),
        undersampler=ROSE()),
    measures=(SX.accuracy, )
)

# ---------------------------------------------------------------------------- #
#                           balancing with tuning                              #
# ---------------------------------------------------------------------------- #
r1 = SX.range(:(oversampler.k), lower=3, upper=10)
r2 = SX.range(:(undersampler.min_ratios), lower=0.1, upper=0.9)

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2)),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.RandomForestClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2)),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2)),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.ModalDecisionTree(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2)),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.ModalRandomForest(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2)),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=ModalAdaBoost(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2)),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = symbolic_analysis(
    Xc, yc;
    model=SX.XGBoostClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2)),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

@test_throws ArgumentError symbolic_analysis(
    Xr, yr;
    model=SX.XGBoostRegressor(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2)),
    measures=(SX.accuracy, )
)

# ds = setup_dataset(
#     Xc, yc;
#     model=SX.XGBoostClassifier(),
#     resampling=CV(nfolds=5, shuffle=true),
#     seed=11,
#     balancing=(
#         oversampler=SMOTENC(k=5, ratios=1.0),
#         undersampler=TomekUndersampler(min_ratios=0.5)),
#     tuning=GridTuning(goal=4, range=(r1,r2)),
# )
# ds.pidxs[1].test
# train = get_train(ds.pidxs[1])
# MLJ.fit!(ds.mach, rows=train, verbosity=0)
# m=ds.mach

# ---------------------------------------------------------------------------- #
#                                   tuning                                     #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = symbolic_analysis(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = symbolic_analysis(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    tuning=RandomTuning(;range, resampling=CV(nfolds=3), measure=SX.accuracy, repeats=2),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = symbolic_analysis(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    tuning=RandomTuning(;range, resampling=CV(nfolds=3), measure=SX.accuracy, repeats=2),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = symbolic_analysis(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    tuning=ParticleTuning(resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = symbolic_analysis(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    tuning=AdaptiveTuning(resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

# selector = FeatureSelector()
# range = MLJ.range(selector, :features, values = ((:sepal_length,), (:sepal_length, :sepal_width)))
# iterator(r2)

# ---------------------------------------------------------------------------- #
#                              instance weights                                #
# ---------------------------------------------------------------------------- #
wc = rand(length(yc))
modelw = symbolic_analysis(
    Xc, yc, wc;
    model=SX.XGBoostClassifier(),
    seed=1,
    measures=(SX.accuracy,)
)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                                  windowing                                   #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    measures=(SX.accuracy, log_loss)      
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    win=WholeWindow(),
    measures=(SX.accuracy, log_loss)      
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    win=SplitWindow(nwindows=2),
    measures=(SX.accuracy, log_loss)      
)
@test modelts isa SX.ModelSet

modelts = symbolic_analysis(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    win=MovingWindow(window_size=20, window_step=5),
    measures=(SX.accuracy, log_loss)      
)
@test modelts isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                               get_operations                                 #
# ---------------------------------------------------------------------------- #
@testset "probabilistic predictions" begin
    # mock measures with different properties
    struct MockMeasure{K, O}
        kind_of_proxy::K
        observation_scitype::O
    end
    
    # override MLJ methods for our mock measures
    MLJ.MLJBase.StatisticalMeasuresBase.kind_of_proxy(m::MockMeasure) = m.kind_of_proxy
    MLJ.MLJBase.StatisticalMeasuresBase.observation_scitype(m::MockMeasure) = m.observation_scitype
    
    # invalid
    point_ambiguous = MockMeasure(MLJ.MLJBase.LearnAPI.Point(), String)
    @test_throws Exception SX.get_operations([point_ambiguous], :probabilistic)
    
    # test unknown proxy type
    unknown_proxy = MockMeasure("UnknownProxy", Missing)
    @test_throws Exception SX.get_operations([unknown_proxy], :probabilistic)
end

@testset "deterministic predictions" begin    
    # test Distribution proxy (should throw error)
    dist_measure = MockMeasure(MLJ.MLJBase.LearnAPI.Distribution(), Missing)
    @test_throws Exception SX.get_operations([dist_measure], :deterministic)
    
    # test unknown proxy type
    unknown_proxy = MockMeasure("UnknownProxy", Missing)
    @test_throws Exception SX.get_operations([unknown_proxy], :deterministic)
end

@testset "interval predictions" begin
    # test ConfidenceInterval proxy
    interval_measure = MockMeasure(MLJ.MLJBase.LearnAPI.ConfidenceInterval(), Missing)
    ops = SX.get_operations([interval_measure], :interval)
    @test ops[1] == SX.sole_predict
    
    # test non-ConfidenceInterval proxy
    point_measure = MockMeasure(MLJ.MLJBase.LearnAPI.Point(), Missing)
    @test_throws Exception SX.get_operations([point_measure], :interval)
end

@testset "unsupported prediction types" begin
    point_measure = MockMeasure(MLJ.MLJBase.LearnAPI.Point(), Missing)
    @test_throws Exception SX.get_operations([point_measure], :unsupported)
    @test_throws Exception SX.get_operations([point_measure], :invalid)
end

# ---------------------------------------------------------------------------- #
#                                   base.show                                  #
# ---------------------------------------------------------------------------- #
@testset "Base.show methods" begin
    modelc = symbolic_analysis(Xc, yc)
    # Test Measures show methods
    @testset "Measures show" begin

        output = string(modelc.measures)
        @test contains(output, "Measures(")
        @test contains(output, "Accuracy")
        @test contains(output, "Kappa")

        # Test pretty show
        io = IOBuffer()
        show(io, MIME"text/plain"(), modelc.measures)
        output = String(take!(io))
        @test contains(output, "Measures:")
    end
    
    @testset "ModelSet show" begin        
        # Test basic output
        output = string(modelc)
        @test contains(output, "ModelSet")
        @test contains(output, "models=")
        
        # Test pretty show
        io = IOBuffer()
        show(io, MIME"text/plain"(), modelc)
        output = String(take!(io))
        @test contains(output, "Dataset:")
        @test contains(output, "Models:")
    end
end

# ---------------------------------------------------------------------------- #
#                               X not a dataframe                              #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
modelc = symbolic_analysis(Xc, yc)
@test modelc isa SX.ModelSet
