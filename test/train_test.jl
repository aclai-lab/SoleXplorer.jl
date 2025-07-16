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

# ---------------------------------------------------------------------------- #
#                        train and test usage examples                         #
# ---------------------------------------------------------------------------- #
# basic setup
solemc = train_test(Xc, yc)
@test solemc isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeClassifier}}
solemr = train_test(Xr, yr)
@test solemr isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeRegressor}}

datac  = setup_dataset(Xc, yc)
solemc = train_test(datac)
@test solemc isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeClassifier}}
datar  = setup_dataset(Xr, yr)
solemr = train_test(datar)
@test solemr isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeRegressor}}

# ---------------------------------------------------------------------------- #
#                                     models                                   #
# ---------------------------------------------------------------------------- #
solemc = train_test(
    Xc, yc;
    model=DecisionTreeClassifier()
)
@test solemc isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeClassifier}}

solemc = train_test(
    Xc, yc;
    model=RandomForestClassifier()
)
@test solemc isa SX.ModelSet{SX.PropositionalDataSet{RandomForestClassifier}}

solemc = train_test(
    Xc, yc;
    model=AdaBoostStumpClassifier()
)
@test solemc isa SX.ModelSet{SX.PropositionalDataSet{AdaBoostStumpClassifier}}

solemr = train_test(
    Xr, yr;
    model=DecisionTreeRegressor()
)
@test solemr isa SX.ModelSet{SX.PropositionalDataSet{DecisionTreeRegressor}}

solemr = train_test(
    Xr, yr;
    model=RandomForestRegressor()
)
@test solemr isa SX.ModelSet{SX.PropositionalDataSet{RandomForestRegressor}}

solemts = train_test(
    Xts, yts;
    model=ModalDecisionTree()
)
@test solemts isa SX.ModelSet{SX.ModalDataSet{ModalDecisionTree}}

solemts = train_test(
    Xts, yts;
    model=ModalRandomForest()
)
@test solemts isa SX.ModelSet{SX.ModalDataSet{ModalRandomForest}}

solemts = train_test(
    Xts, yts;
    model=ModalAdaBoost()
)
@test solemts isa SX.ModelSet{SX.ModalDataSet{ModalAdaBoost}}

solemc = train_test(
    Xc, yc;
    model=XGBoostClassifier()
)
@test solemc isa SX.ModelSet{SX.PropositionalDataSet{XGBoostClassifier}}

solemr = train_test(
    Xr, yr;
    model=XGBoostRegressor()
)
@test solemr isa SX.ModelSet{SX.PropositionalDataSet{XGBoostRegressor}}

# ---------------------------------------------------------------------------- #
#                                     tuning                                   #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemc = train_test(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemc isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:DecisionTreeClassifier}}}

range = (
    SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log),
    SX.range(:n_trees; lower=10, unit=20, upper=90)
)
solemc = train_test(
    Xc, yc;
    model=RandomForestClassifier(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemc isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:RandomForestClassifier}}}

range = SX.range(:n_iter; lower=10, unit=10, upper=100)
solemc = train_test(
    Xc, yc;
    model=AdaBoostStumpClassifier(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemc isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:AdaBoostStumpClassifier}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemr = train_test(
    Xr, yr;
    model=DecisionTreeRegressor(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)
)
@test solemr isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:DecisionTreeRegressor}}}

range = (
    SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log),
    SX.range(:n_trees; lower=10, unit=20, upper=90)
)
solemr = train_test(
    Xr, yr;
    model=RandomForestRegressor(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)
)
@test solemr isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:RandomForestRegressor}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemts = train_test(
    Xts, yts;
    model=ModalDecisionTree(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemts isa SX.ModelSet{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:ModalDecisionTree}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemts = train_test(
    Xts, yts;
    model=ModalRandomForest(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemts isa SX.ModelSet{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:ModalRandomForest}}}

range = SX.range(:n_iter; lower=2, unit=10, upper=10)
solemts = train_test(
    Xts, yts;
    model=ModalAdaBoost(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemts isa SX.ModelSet{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:ModalAdaBoost}}}

range = SX.range(:num_round; lower=10, unit=10, upper=100)
solemc = train_test(
    Xc, yc;
    model=XGBoostClassifier(
        early_stopping_rounds=20,
    ),
    resample=(type=CV(nfolds=5, shuffle=true), valid_ratio=0.2, rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemc isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:XGBoostClassifier}}}

range = SX.range(:num_round; lower=10, unit=10, upper=100)
solemr = train_test(
    Xr, yr;
    model=XGBoostRegressor(
        early_stopping_rounds=20,
    ),
    resample=(type=CV(nfolds=5, shuffle=true), valid_ratio=0.2, rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)
)
@test solemr isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:XGBoostRegressor}}}
