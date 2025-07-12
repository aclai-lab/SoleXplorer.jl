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
#                        prepare dataset usage examples                        #
# ---------------------------------------------------------------------------- #
# basic setup
modelc = prepare_dataset(Xc, yc)
@test modelc isa SoleXplorer.PropositionalDataSet{DecisionTreeClassifier}
modelr = prepare_dataset(Xr, yr)
@test modelr isa SoleXplorer.PropositionalDataSet{DecisionTreeRegressor}


# model type specification
modelc = prepare_dataset(
    Xc, yc;
    model=DecisionTreeClassifier()
)
@test modelc isa SoleXplorer.PropositionalDataSet{DecisionTreeClassifier}

modelc = prepare_dataset(
    Xc, yc;
    model=RandomForestClassifier()
)
@test modelc isa SoleXplorer.PropositionalDataSet{RandomForestClassifier}

modelc = prepare_dataset(
    Xc, yc;
    model=AdaBoostStumpClassifier()
)
@test modelc isa SoleXplorer.PropositionalDataSet{AdaBoostStumpClassifier}

modelr = prepare_dataset(
    Xr, yr;
    model=DecisionTreeRegressor()
)
@test modelr isa SoleXplorer.PropositionalDataSet{DecisionTreeRegressor}

modelr = prepare_dataset(
    Xr, yr;
    model=RandomForestRegressor()
)
@test modelr isa SoleXplorer.PropositionalDataSet{RandomForestRegressor}

modelc = prepare_dataset(
    Xts, yts;
    model=ModalDecisionTree()
)
@test modelc isa SoleXplorer.ModalDataSet{ModalDecisionTree}

modelc = prepare_dataset(
    Xts, yts;
    model=ModalRandomForest()
)
@test modelc isa SoleXplorer.ModalDataSet{ModalRandomForest}

modelc = prepare_dataset(
    Xts, yts;
    model=ModalAdaBoost()
)
@test modelc isa SoleXplorer.ModalDataSet{ModalAdaBoost}

modelc = prepare_dataset(
    Xc, yc;
    model=XGBoostClassifier()
)
@test modelc isa SoleXplorer.PropositionalDataSet{XGBoostClassifier}

modelr = prepare_dataset(
    Xr, yr;
    model=XGBoostRegressor()
)
@test modelr isa SoleXplorer.PropositionalDataSet{XGBoostRegressor}

# ---------------------------------------------------------------------------- #
#                covering various examples to complete codecov                 #
# ---------------------------------------------------------------------------- #
y_symbol = :petal_width
modelc = prepare_dataset(Xc, y_symbol)
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

# dataset is composed also of non numeric columns
Xnn = hcat(Xc, DataFrame(target = yc))
@test_nowarn SX.code_dataset(Xnn)

modelts, dts = prepare_dataset(Xts, yts)

modelc = prepare_dataset(
    Xc, yc;
    preprocess=(train_ratio=0.5, vnames=["p1", "p2", "p3", "p4"], modalreduce=maximum)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

# ---------------------------------------------------------------------------- #
#                                 resamplig                                    #
# ---------------------------------------------------------------------------- #
modelc = prepare_dataset(
    Xc, yc;
    resample=(;type=CV)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc = prepare_dataset(
    Xc, yc;
    resample=(;type=Holdout)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc = prepare_dataset(
    Xc, yc;
    resample=(;type=StratifiedCV)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc = prepare_dataset(
    Xc, yc;
    resample=(;type=TimeSeriesCV)
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

modelc = prepare_dataset(
    Xc, yc;
    resample=(type=CV, params=(nfolds=10, shuffle=true, rng=rng=Xoshiro(1)))
)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset

# ---------------------------------------------------------------------------- #
#                            validate modelsetup                               #
# ---------------------------------------------------------------------------- #
modelc = prepare_dataset(
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

@test_throws ArgumentError prepare_dataset(
    Xc, yc;
    preprocess=(train_ratio=0.5, invalid=maximum)
)

@test_throws ArgumentError prepare_dataset(
    Xc, yc;
    preprocess=(;vnames=[:p1, :p2, :p3, :p4, :p5])
)

Xrc, yrc = SoleData.load_arff_dataset("NATOPS")
Xrc[1,1] = Xrc[1,1][1:end-4]
Xrc[1,2] = Xrc[1,1][1:end-3]

@test_throws ArgumentError prepare_dataset(
    Xrc, yrc;
    model=(;type=:modaldecisiontree)
)

# ---------------------------------------------------------------------------- #
#                                dataset info                                  #
# ---------------------------------------------------------------------------- #
modelc = prepare_dataset(Xc, yc)

@test SX.get_treatment(dsc.info) == :aggregate
@test SX.get_modalreduce(dsc.info) == mean
@test SX.get_train_ratio(dsc.info) == 0.7
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

# ---------------------------------------------------------------------------- #
#                         check vnames and modalreduce                         #
# ---------------------------------------------------------------------------- #
vnames=[:p1, :p2, :p3, :p4]
modelc, _ = prepare_dataset(
    Xc, yc;
    preprocess=(;vnames)
)
@test modelc.setup.preprocess.vnames == vnames

_, dsmin = prepare_dataset(Xts, yts; model=(;type=:modaldecisiontree), preprocess=(;modalreduce=minimum))
_, dsmax = prepare_dataset(Xts, yts; model=(;type=:modaldecisiontree), preprocess=(;modalreduce=maximum))
@test all(dsmin.X .<= dsmax.X)

# ---------------------------------------------------------------------------- #
#                                  windowing                                   #
# ---------------------------------------------------------------------------- #
modelts, dts = prepare_dataset(
    Xts, yts;
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1))
)
@test modelts isa SoleXplorer.Modelset
@test dts    isa SoleXplorer.Dataset