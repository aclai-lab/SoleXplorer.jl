using Test
using SoleXplorer
const SX = SoleXplorer

using DataTreatments
const DT = DataTreatments

using MLJ
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SX.NatopsLoader()
Xts, yts = SX.load(natopsloader)

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

function build_2_bananas_df()
    DataFrame(
        str_col=[missing, "blue", "green", "red", "blue"],
        sym_col=[:circle, :square, :triangle, :square, missing],
        img4=[i == 3 ? missing : create_image(i + 30) for i in 1:5],
        int_col=Int[10, 20, 30, 40, 50],
        V1=[NaN, missing, 3.0, 4.0, 5.6],
        V2=[2.5, missing, 4.5, 5.5, NaN],
        ts1=[
            NaN, collect(2.0:7.0),
            missing, collect(4.0:9.0),
            collect(5.0:10.0)
        ],
        V4=[4.1, NaN, NaN, 7.1, 5.5],
        V5=[5.0, 6.0, 7.0, 8.0, 1.8],
        ts2=[
            collect(2.0:0.5:5.5),
            collect(1.0:0.5:4.5),
            collect(3.0:0.5:6.5),
            collect(4.0:0.5:7.5),
            NaN
        ],
        ts3=[
            [1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2],
            NaN, NaN, missing,
            [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]
        ],
        V3=[3.2, 4.2, 5.2, missing, 2.4],
        ts4=[
            [6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8],
            missing,
            [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8],
            [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8],
            [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]
        ],
        img1=[create_image(i) for i in 1:5],
        cat_col=categorical(["small", "medium", missing, "small", "large"]),
        uint_col=UInt32[1, 2, 3, 4, 5],
        img2=[i == 1 ? NaN : create_image(i + 10) for i in 1:5],
        img3=[create_image(i + 20) for i in 1:5],
    )
end

function build_test_df()
    n = 20
    Random.seed!(42)
    DataFrame(
        str_col=rand([missing, "blue", "green", "red"], n),
        sym_col=rand([:circle, :square, :triangle, missing], n),
        img4=[i % 5 == 0 ? missing : create_image(i + 30) for i in 1:n],
        int_col=Int.(collect(10:10:10*n)),
        V1=collect(1.0:Float64(n)),
        V2=collect(1.0:Float64(n)) .* 2.5,
        ts1=[collect(Float64(i):Float64(i)+5) for i in 1:n],
        V4=collect(4.0:0.5:4.0+0.5*(n-1)),
        V5=collect(5.0:0.5:5.0+0.5*(n-1)),
        ts2=[collect(Float64(i):0.5:Float64(i)+3.5) for i in 1:n],
        ts3=[collect(Float64(i):0.2:Float64(i)+1.2) for i in 1:n],
        V3=collect(3.0:0.5:3.0+0.5*(n-1)),
        ts4=[collect(Float64(i)+6:-0.8:Float64(i)) for i in 1:n],
        img1=[create_image(i) for i in 1:n],
        cat_col=categorical(rand(["small", "medium", "large"], n)),
        uint_col=UInt32.(1:n),
        img2=[create_image(i + 10) for i in 1:n],
        img3=[create_image(i + 20) for i in 1:n],
    )
end

d_bananas = build_2_bananas_df()
tb_classif = ["classA", "classB", "classC", "classA", "classB"]
tb_regress = [1.0, 5.0, 10.0, 15.0, 20.0]

df = build_test_df()
t_classif = repeat(["classA", "classB", "classC", "classA", "classB"], 4)
t_regress = collect(1.0:1.0:20.0)  

# ---------------------------------------------------------------------------- #
#                        I'm easy like sunday morning                          #
# ---------------------------------------------------------------------------- #
modelc = solexplorer(Xc, yc)
@test modelc isa SX.ModelSet

modelr = solexplorer(Xr, yr)
@test modelc isa SX.ModelSet

modelts = solexplorer(Xts, yts)
@test modelc isa SX.ModelSet

modeldb = solexplorer(d_bananas, tb_classif, 
    TreatmentGroup(impute=(LOCF(), NOCB()))
)
@test modelc isa SX.ModelSet

modeldb = solexplorer(df, t_regress,
    TreatmentGroup(impute=(LOCF(), NOCB()))
)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                               usage examples                                 #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
tuning=GridTuning(;
    range,
    resolution=10,
    resampling=CV(nfolds=3),
    measure=SX.accuracy,
    repeats=2
)
modelc = solexplorer(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=42,
    tuning,
    extractor=SX.InTreesRuleExtractor(),
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
tuning=GridTuning(resolution=20, resampling=CV(nfolds=3), range=range, repeats=2) 
modelr = solexplorer(
    Xr, yr;
    model=SX.DecisionTreeRegressor(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=42,
    tuning,
    measures=(rms, l1, l2, mae, mav)
)
@test modelr isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                               usage example #2                               #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = solexplorer(
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
modelr = solexplorer(
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
modelc = solexplorer(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    resampling=Holdout(fraction_train=0.75, shuffle=true),
    seed=1,
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelc isa SX.ModelSet

modelr = solexplorer(
    Xr, yr;
    model=SX.RandomForestRegressor(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    measures=(rms, l1, l2, mae, mav)      
)
@test modelr isa SX.ModelSet

modelc = solexplorer(
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
modelts = solexplorer(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    resampling=Holdout(fraction_train=0.5, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
    reducefunc=mean,
    features=(maximum, minimum),
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet

modelts = solexplorer(
    Xts, yts;
    model=SX.RandomForestClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
    reducefunc=mean,
    features=(maximum, minimum),
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet

modelts = solexplorer(
    Xts, yts;
    model=SX.AdaBoostStumpClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
    reducefunc=mean,
    features=(maximum, minimum),
    measures=(SX.accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet

# TODO known bug, see TODO.md 
# modelts = solexplorer(
#     Xts, yts;
#     model=SX.XGBoostClassifier(),
#     resampling=(type=TimeSeriesCV(nfolds=5), seed=1),
#     win=adaptivewindow(nwindows=3, overlap=0.3),
#     reducefunc=mean,
#     features=(maximum, minimum),
#     measures=(SX.accuracy, log_loss, confusion_matrix, kappa)
# )
# @test modelts isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                       resampling in modal time series                        #
# ---------------------------------------------------------------------------- #
modelts = solexplorer(
    Xts, yts;
    model=SX.ModalDecisionTree(),
    resampling=CV(;nfolds=4),
    seed=1,
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

modelts = solexplorer(
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
modelc = solexplorer(
    Xc, yc;
    model=SX.XGBoostClassifier(early_stopping_rounds=20),
    resampling=CV(nfolds=5, shuffle=true),
    valid_ratio=0.2,
    seed=1,
    measures=(confusion_matrix,) 
)
@test modelc isa SX.ModelSet

range = SX.range(:num_round; lower=10, unit=10, upper=100)
modelr = solexplorer(
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
modelts = solexplorer(
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

modelts = solexplorer(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    features=(base_set...,),
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

modelts = solexplorer(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    features=(catch9...,),
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

modelts = solexplorer(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    features=(catch22_set...,),
    measures=(SX.accuracy,)
)
@test modelts isa SX.ModelSet

modelts = solexplorer(
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
modelc = solexplorer(
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

modelc = solexplorer(
    Xc, yc;
    model=SX.RandomForestClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=ENNUndersampler(k=7),
        undersampler=ROSE()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = solexplorer(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=RandomOversampler(),
        undersampler=RandomUndersampler()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = solexplorer(
    Xc, yc;
    model=SX.ModalDecisionTree(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=RandomWalkOversampler(),
        undersampler=SMOTE()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = solexplorer(
    Xc, yc;
    model=SX.ModalRandomForest(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTE(),
        undersampler=RandomUndersampler()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = solexplorer(
    Xc, yc;
    model=ModalAdaBoost(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTENC(),
        undersampler=TomekUndersampler()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

modelc = solexplorer(
    Xc, yc;
    model=SX.XGBoostClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTENC(),
        undersampler=TomekUndersampler()),
    measures=(SX.accuracy, )
)
@test modelc isa SX.ModelSet

@test_throws ArgumentError solexplorer(
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

modelc = solexplorer(
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

modelc = solexplorer(
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

modelc = solexplorer(
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

modelc = solexplorer(
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

modelc = solexplorer(
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

modelc = solexplorer(
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

modelc = solexplorer(
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

@test_throws ArgumentError solexplorer(
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
modelc = solexplorer(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = solexplorer(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    tuning=RandomTuning(;range, resampling=CV(nfolds=3), measure=SX.accuracy, repeats=2),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = solexplorer(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    tuning=RandomTuning(;range, resampling=CV(nfolds=3), measure=SX.accuracy, repeats=2),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = solexplorer(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    tuning=ParticleTuning(resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2),
    measures=(log_loss, SX.accuracy, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = solexplorer(
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
modelw = solexplorer(
    Xc, yc, wc;
    model=SX.XGBoostClassifier(),
    seed=1,
    measures=(SX.accuracy,)
)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                                  windowing                                   #
# ---------------------------------------------------------------------------- #
modelts = solexplorer(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
    measures=(SX.accuracy, log_loss)      
)
@test modelts isa SX.ModelSet

modelts = solexplorer(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    win=wholewindow(),
    measures=(SX.accuracy, log_loss)      
)
@test modelts isa SX.ModelSet

modelts = solexplorer(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    win=splitwindow(nwindows=2),
    measures=(SX.accuracy, log_loss)      
)
@test modelts isa SX.ModelSet

modelts = solexplorer(
    Xts, yts;
    model=SX.DecisionTreeClassifier(),
    seed=1,
    win=movingwindow(winsize=20, winstep=5),
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
    modelc = solexplorer(Xc, yc)
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
modelc = solexplorer(Xc, yc)
@test modelc isa SX.ModelSet
