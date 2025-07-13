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
modelc = prepare_dataset(
    Xc, yc;
    model=DecisionTreeClassifier()
)
@test modelc isa SX.PropositionalDataSet{DecisionTreeClassifier}

modelc = prepare_dataset(
    Xc, yc;
    model=RandomForestClassifier()
)
@test modelc isa SX.PropositionalDataSet{RandomForestClassifier}

modelc = prepare_dataset(
    Xc, yc;
    model=AdaBoostStumpClassifier()
)
@test modelc isa SX.PropositionalDataSet{AdaBoostStumpClassifier}

modelr = prepare_dataset(
    Xr, yr;
    model=DecisionTreeRegressor()
)
@test modelr isa SX.PropositionalDataSet{DecisionTreeRegressor}

modelr = prepare_dataset(
    Xr, yr;
    model=RandomForestRegressor()
)
@test modelr isa SX.PropositionalDataSet{RandomForestRegressor}

modelts = prepare_dataset(
    Xts, yts;
    model=ModalDecisionTree()
)
@test modelc isa SX.ModalDataSet{ModalDecisionTree}

modelts = prepare_dataset(
    Xts, yts;
    model=ModalRandomForest()
)
@test modelc isa SX.ModalDataSet{ModalRandomForest}

modelts = prepare_dataset(
    Xts, yts;
    model=ModalAdaBoost()
)
@test modelc isa SX.ModalDataSet{ModalAdaBoost}

modelc = prepare_dataset(
    Xc, yc;
    model=XGBoostClassifier()
)
@test modelc isa SX.PropositionalDataSet{XGBoostClassifier}

modelr = prepare_dataset(
    Xr, yr;
    model=XGBoostRegressor()
)
@test modelr isa SX.PropositionalDataSet{XGBoostRegressor}

# ---------------------------------------------------------------------------- #
#                                     tuning                                   #
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


