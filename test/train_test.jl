using Test
using MLJ, SoleXplorer
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

# ---------------------------------------------------------------------------- #
#                        train and test usage examples                         #
# ---------------------------------------------------------------------------- #
# basic setup
modelc, _, _ = train_test(Xc, yc)
@test modelc isa SoleXplorer.Modelset
modelr, _, _ = train_test(Xr, yr)
@test modelr isa SoleXplorer.Modelset

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
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ),
    preprocess=(;rng=Xoshiro(1))
)
@test modelc isa SoleXplorer.Modelset

modelc, _, _ = train_test(
    Xc, yc;
    model=(;type=:randomforest),
    tuning=(
        method=(;type=randomsearch),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ),
)
@test modelc isa SoleXplorer.Modelset

modelc, _, _ = train_test(
    Xc, yc;
    model=(;type=:adaboost),
    tuning=(
        method=(;type=latinhypercube),
        ranges=(SoleXplorer.range(:n_iter, lower=2, upper=10),)
    ),
)
@test modelc isa SoleXplorer.Modelset

modelr, _, _ = train_test(
    Xr, yr;
    model=(;type=:decisiontree),
        tuning=(
        method=(;type=particleswarm),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ),
)
@test modelr isa SoleXplorer.Modelset

modelr, _, _ = train_test(
    Xr, yr;
    model=(;type=:randomforest),
    tuning=(
        method=(;type=adaptiveparticleswarm),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ),
)
@test modelr isa SoleXplorer.Modelset
@test modeltype(modelr) == SX.AbstractRegression

# ---------------------------------------------------------------------------- #
#                          modelsetup and modelset                             #
# ---------------------------------------------------------------------------- #
modelc, _, _ = train_test(Xc, yc)
@test modelc isa SoleXplorer.Modelset

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


