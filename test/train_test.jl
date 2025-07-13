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

# TODO propaga l'rng nel tuning

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
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy)
)
@test modelc isa SX.ModelSet{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:DecisionTreeClassifier}}}

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
# model type specification
modelc, _, _ = train_test(
    Xc, yc;
    model=(;type=:decisiontree),
    tuning=(
        method=(;type=grid),
        ranges=(
            SX.range(:max_depth, lower=2, upper=10),
            SX.range(:feature_importance, values=[:impurity, :split])
        )
    ),
    preprocess=(;rng=Xoshiro(1))
)
@test modelc isa SX.Modelset

modelc, _, _ = train_test(
    Xc, yc;
    model=(;type=:randomforest),
    tuning=(
        method=(;type=randomsearch),
        ranges=(
            SX.range(:max_depth, lower=2, upper=10),
            SX.range(:feature_importance, values=[:impurity, :split])
        )
    ),
)
@test modelc isa SX.Modelset

modelc, _, _ = train_test(
    Xc, yc;
    model=(;type=:adaboost),
    tuning=(
        method=(;type=latinhypercube),
        ranges=(SX.range(:n_iter, lower=2, upper=10),)
    ),
)
@test modelc isa SX.Modelset

modelr, _, _ = train_test(
    Xr, yr;
    model=(;type=:decisiontree),
        tuning=(
        method=(;type=particleswarm),
        ranges=(
            SX.range(:max_depth, lower=2, upper=10),
            SX.range(:feature_importance, values=[:impurity, :split])
        )
    ),
)
@test modelr isa SX.Modelset

modelr, _, _ = train_test(
    Xr, yr;
    model=(;type=:randomforest),
    tuning=(
        method=(;type=adaptiveparticleswarm),
        ranges=(
            SX.range(:max_depth, lower=2, upper=10),
            SX.range(:feature_importance, values=[:impurity, :split])
        )
    ),
)
@test modelr isa SX.Modelset
@test modeltype(modelr) == SX.AbstractRegression

# ---------------------------------------------------------------------------- #
#                          modelsetup and modelset                             #
# ---------------------------------------------------------------------------- #
modelc, _, _ = train_test(Xc, yc)
@test modelc isa SX.Modelset

@test SX.get_resample(modelc.setup) isa SX.Resample
@test SX.get_resultsparams(modelc.setup) isa Function
@test_nowarn sprint(show, modelc.setup)

@test SX.DecisionTreeClassifierModel(modelc.setup) isa SX.ModelSetup
@test SX.RandomForestClassifierModel(modelc.setup) isa SX.ModelSetup
@test SX.AdaBoostClassifierModel(modelc.setup) isa SX.ModelSetup
@test SX.DecisionTreeRegressorModel(modelc.setup) isa SX.ModelSetup
@test SX.RandomForestRegressorModel(modelc.setup) isa SX.ModelSetup
@test SX.ModalDecisionTreeModel(modelc.setup) isa SX.ModelSetup
@test SX.ModalRandomForestModel(modelc.setup) isa SX.ModelSetup
@test SX.ModalAdaBoostModel(modelc.setup) isa SX.ModelSetup
@test SX.XGBoostClassifierModel(modelc.setup) isa SX.ModelSetup
@test SX.XGBoostRegressorModel(modelc.setup) isa SX.ModelSetup

@test_nowarn sprint(show, modelc)


