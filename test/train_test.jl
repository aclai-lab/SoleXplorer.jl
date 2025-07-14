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
#                        train and test usage examples                         #
# ---------------------------------------------------------------------------- #
# basic setup
modelc = train_test(Xc, yc)
@test modelc isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeClassifier}}
modelr = train_test(Xr, yr)
@test modelr isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeRegressor}}

datac  = prepare_dataset(Xc, yc)
modelc = train_test(datac)
@test modelc isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeClassifier}}
datar  = prepare_dataset(Xr, yr)
modelr = train_test(datar)
@test modelr isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeRegressor}}

# ---------------------------------------------------------------------------- #
#                                     models                                   #
# ---------------------------------------------------------------------------- #
modelc = train_test(
    Xc, yc;
    model=DecisionTreeClassifier()
)
@test modelc isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeClassifier}}

modelc = train_test(
    Xc, yc;
    model=RandomForestClassifier()
)
@test modelc isa SX.ModelSet{SX.PropositionalDataSet{RandomForestClassifier}}

modelc = train_test(
    Xc, yc;
    model=AdaBoostStumpClassifier()
)
@test modelc isa SX.ModelSet{SX.PropositionalDataSet{AdaBoostStumpClassifier}}

modelr = train_test(
    Xr, yr;
    model=DecisionTreeRegressor()
)
@test modelr isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeRegressor}}

modelr = train_test(
    Xr, yr;
    model=RandomForestRegressor()
)
@test modelr isa SX.ModelSet{SX.PropositionalDataSet{RandomForestRegressor}}

modelts = train_test(
    Xts, yts;
    model=ModalDecisionTree()
)
@test modelts isa SX.ModelSet{SX.ModalDataSet{ModalDecisionTree}}

modelts = train_test(
    Xts, yts;
    model=ModalRandomForest()
)
@test modelts isa SX.ModelSet{SX.ModalDataSet{ModalRandomForest}}

modelts = train_test(
    Xts, yts;
    model=ModalAdaBoost()
)
@test modelts isa SX.ModelSet{SX.ModalDataSet{ModalAdaBoost}}

modelc = train_test(
    Xc, yc;
    model=XGBoostClassifier()
)
@test modelc isa SX.ModelSet{SX.PropositionalDataSet{XGBoostClassifier}}

modelr = train_test(
    Xr, yr;
    model=XGBoostRegressor()
)
@test modelr isa SX.ModelSet{SX.PropositionalDataSet{XGBoostRegressor}}

# ---------------------------------------------------------------------------- #
#                                     tuning                                   #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = train_test(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test modelc isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:DecisionTreeClassifier}}}

range = (
    SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log),
    SX.range(:n_trees; lower=10, unit=20, upper=90)
)
modelc = train_test(
    Xc, yc;
    model=RandomForestClassifier(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test modelc isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:RandomForestClassifier}}}

range = SX.range(:n_iter; lower=10, unit=10, upper=100)
modelc = train_test(
    Xc, yc;
    model=AdaBoostStumpClassifier(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test modelc isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:AdaBoostStumpClassifier}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelr = train_test(
    Xr, yr;
    model=DecisionTreeRegressor(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)
)
@test modelr isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:DecisionTreeRegressor}}}

range = (
    SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log),
    SX.range(:n_trees; lower=10, unit=20, upper=90)
)
modelr = train_test(
    Xr, yr;
    model=RandomForestRegressor(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)
)
@test modelr isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:RandomForestRegressor}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelts = train_test(
    Xts, yts;
    model=ModalDecisionTree(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test modelts isa SX.ModelSet{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:ModalDecisionTree}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelts = train_test(
    Xts, yts;
    model=ModalRandomForest(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test modelts isa SX.ModelSet{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:ModalRandomForest}}}

range = SX.range(:n_iter; lower=2, unit=10, upper=10)
modelts = train_test(
    Xts, yts;
    model=ModalAdaBoost(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test modelts isa SX.ModelSet{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:ModalAdaBoost}}}

range = SX.range(:num_round; lower=10, unit=10, upper=100)
modelc = train_test(
    Xc, yc;
    model=XGBoostClassifier(
        early_stopping_rounds=20,
    ),
    resample=(type=CV(nfolds=5, shuffle=true), valid_ratio=0.2, rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test modelc isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:XGBoostClassifier}}}

range = SX.range(:num_round; lower=10, unit=10, upper=100)
modelr = train_test(
    Xr, yr;
    model=XGBoostRegressor(
        early_stopping_rounds=20,
    ),
    resample=(type=CV(nfolds=5, shuffle=true), valid_ratio=0.2, rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)
)
@test modelr isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:XGBoostRegressor}}}

# ---------------------------------------------------------------------------- #
#                          modelsetup and modelset                             #
# ---------------------------------------------------------------------------- #
# modelc, _, _ = train_test(Xc, yc)
# @test modelc isa SX.Modelset

# @test SX.get_resample(modelc.setup) isa SX.Resample
# @test SX.get_resultsparams(modelc.setup) isa Function
# @test_nowarn sprint(show, modelc.setup)

# @test SX.DecisionTreeClassifierModel(modelc.setup) isa SX.ModelSetup
# @test SX.RandomForestClassifierModel(modelc.setup) isa SX.ModelSetup
# @test SX.AdaBoostClassifierModel(modelc.setup) isa SX.ModelSetup
# @test SX.DecisionTreeRegressorModel(modelc.setup) isa SX.ModelSetup
# @test SX.RandomForestRegressorModel(modelc.setup) isa SX.ModelSetup
# @test SX.ModalDecisionTreeModel(modelc.setup) isa SX.ModelSetup
# @test SX.ModalRandomForestModel(modelc.setup) isa SX.ModelSetup
# @test SX.ModalAdaBoostModel(modelc.setup) isa SX.ModelSetup
# @test SX.XGBoostClassifierModel(modelc.setup) isa SX.ModelSetup
# @test SX.XGBoostRegressorModel(modelc.setup) isa SX.ModelSetup

# @test_nowarn sprint(show, modelc)

