using Test
using MLJ, SoleXplorer
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

# ---------------------------------------------------------------------------- #
#                        prepare dataset usage examples                        #
# ---------------------------------------------------------------------------- #
# basic setup
modelc, dsc = prepare_dataset(Xc, yc)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset
modelr, dsr = prepare_dataset(Xr, yr)
@test modelr isa SoleXplorer.Modelset
@test dsr    isa SoleXplorer.Dataset

# model type specification
modelc, dsc = prepare_dataset(
    Xc, yc;
    model=(;type=:decisiontree)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    model=(;type=:randomforest)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    model=(;type=:adaboost)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelr, dsr = prepare_dataset(
    Xr, yr;
    model=(;type=:decisiontree)
)
@test modelr isa SoleXplorer.Modelset
@test dsr    isa SoleXplorer.Dataset

modelr, dsr = prepare_dataset(
    Xr, yr;
    model=(;type=:randomforest)
)
@test modelr isa SoleXplorer.Modelset
@test dsr    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    model=(;type=:modaldecisiontree)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    model=(;type=:modalrandomforest)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    model=(;type=:modaladaboost)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    model=(;type=:xgboost)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

# ---------------------------------------------------------------------------- #
#                covering various examples to complete codecov                 #
# ---------------------------------------------------------------------------- #
y_symbol = :petal_width
modelc, dsc = prepare_dataset(Xc, y_symbol)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

# ---------------------------------------------------------------------------- #
#                                 resamplig                                    #
# ---------------------------------------------------------------------------- #
modelc, dsc = prepare_dataset(
    Xc, yc;
    resample=(;type=CV)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    resample=(;type=Holdout)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    resample=(;type=StratifiedCV)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    resample=(;type=TimeSeriesCV)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc, dsc = prepare_dataset(
    Xc, yc;
    resample=(type=CV, params=(nfolds=10, shuffle=true, rng=rng=Xoshiro(1)))
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

# ---------------------------------------------------------------------------- #
#                            validate modelsetup                               #
# ---------------------------------------------------------------------------- #
modelc, dsc = prepare_dataset(
    Xc, yc;
    model=(type=:decisiontree, params=(;max_depth=5))
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

@test_throws ArgumentError prepare_dataset(
    Xc, yc;
    model=(;type=:invalid, params=(;max_depth=5))
)
@test_throws ArgumentError prepare_dataset(
    Xc, yc;
    model=(;params=(;max_depth=5))
)
@test_throws ArgumentError prepare_dataset(
    Xc, yc;
    model=(type=:decisiontree, params=(;invalid=5))
)
