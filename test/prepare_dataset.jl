using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

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

@test_nowarn SX.check_dataset_type(Xc)
@test_nowarn SX.hasnans(Xc)
@test_nowarn SX.code_dataset(Xc)
@test_nowarn SX.code_dataset(yc)
@test_nowarn SX.code_dataset(Xc, yc)
@test_nowarn SX.check_dimensions(Xc)
@test_nowarn SX.check_dimensions(Matrix(Xc))
@test_nowarn SX.find_max_length(Xc)

# modelc, dsc = prepare_dataset(
#     Xc, yc;
#     preprocess=(;vnames=["p1", "p2", "p3", "p4"])
# )

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

# ---------------------------------------------------------------------------- #
#                                dataset info                                  #
# ---------------------------------------------------------------------------- #
modelc, dsc = prepare_dataset(Xc, yc)

@test SX.get_treatment(dsc.info) == :aggregate
@test isnothing(SX.get_reducefunc(dsc.info))
@test SX.get_train_ratio(dsc.info) == 0.8
@test SX.get_valid_ratio(dsc.info) == 0.0
@test SX.get_rng(dsc.info) == TaskLocalRNG()
@test SX.get_vnames(dsc.info) isa Vector{String}
@test_nowarn sprint(show, dsc.info)

output = sprint(show, dsc.info)
@test occursin("DatasetInfo:", output)
@test occursin("treatment:", output)
@test occursin("aggregate", output)

# ---------------------------------------------------------------------------- #
#                                     tt                                       #
# ---------------------------------------------------------------------------- #
@test SX.get_train(dsc.tt[1]) isa Vector{Int64}
@test isempty(SX.get_valid(dsc.tt[1]))
@test SX.get_test(dsc.tt[1]) isa Vector{Int64}
@test length(dsc.tt[1]) == 150
@test_nowarn sprint(show, dsc.tt[1])

# ---------------------------------------------------------------------------- #
#                                   dataset                                    #
# ---------------------------------------------------------------------------- #
@test SX.get_X(dsc) isa Matrix
@test SX.get_y(dsc) isa AbstractVector
@test SX.get_tt(dsc) isa AbstractVector
@test SX.get_info(dsc) isa SX.DatasetInfo
@test_nowarn sprint(show, dsc)

