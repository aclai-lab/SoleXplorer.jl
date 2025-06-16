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
modelc = train_test(Xc, yc)
@test modelc isa SoleXplorer.Modelset
modelr = train_test(Xr, yr)
@test modelr isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                                     tuning                                   #
# ---------------------------------------------------------------------------- #
# model type specification
modelc = train_test(
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

modelc = train_test(
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

modelc = train_test(
    Xc, yc;
    model=(;type=:adaboost),
    tuning=(
        method=(;type=latinhypercube),
        ranges=(SoleXplorer.range(:n_iter, lower=2, upper=10),)
    ),
)
@test modelc isa SoleXplorer.Modelset

modelr = train_test(
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

modelr = train_test(
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

