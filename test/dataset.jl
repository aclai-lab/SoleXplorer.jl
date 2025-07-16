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
#                        prepare dataset usage examples                        #
# ---------------------------------------------------------------------------- #
# basic setup
dsc = setup_dataset(Xc, yc)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
dsr = setup_dataset(Xr, yr)
@test dsr isa SX.PropositionalDataSet{DecisionTreeRegressor}

# model type specification
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier()
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}

dsc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier()
)
@test dsc isa SX.PropositionalDataSet{RandomForestClassifier}

dsc = setup_dataset(
    Xc, yc;
    model=AdaBoostStumpClassifier()
)
@test dsc isa SX.PropositionalDataSet{AdaBoostStumpClassifier}

dsr = setup_dataset(
    Xr, yr;
    model=DecisionTreeRegressor()
)
@test dsr isa SX.PropositionalDataSet{DecisionTreeRegressor}

dsr = setup_dataset(
    Xr, yr;
    model=RandomForestRegressor()
)
@test dsr isa SX.PropositionalDataSet{RandomForestRegressor}

dsts = setup_dataset(
    Xts, yts;
    model=ModalDecisionTree()
)
@test dsts isa SX.ModalDataSet{ModalDecisionTree}

dsts = setup_dataset(
    Xts, yts;
    model=ModalRandomForest()
)
@test dsts isa SX.ModalDataSet{ModalRandomForest}

dsts = setup_dataset(
    Xts, yts;
    model=ModalAdaBoost()
)
@test dsts isa SX.ModalDataSet{ModalAdaBoost}

dsc = setup_dataset(
    Xc, yc;
    model=XGBoostClassifier()
)
@test dsc isa SX.PropositionalDataSet{XGBoostClassifier}

dsr = setup_dataset(
    Xr, yr;
    model=XGBoostRegressor()
)
@test dsr isa SX.PropositionalDataSet{XGBoostRegressor}

# ---------------------------------------------------------------------------- #
#                covering various examples to complete codecov                 #
# ---------------------------------------------------------------------------- #
y_symbol = :petal_width
dsc = setup_dataset(Xc, y_symbol)
@test dsc isa SX.PropositionalDataSet{DecisionTreeRegressor}


# dataset is composed also of non numeric columns
Xnn = hcat(Xc, DataFrame(target = yc))
@test_nowarn SX.code_dataset!(Xnn)

dsc = setup_dataset(
    Xts, yts;
    resample=(train_ratio=0.5,),
    modalreduce=maximum
)
@test dsc isa SX.ModalDataSet{DecisionTreeClassifier}

# ---------------------------------------------------------------------------- #
#                                 resamplig                                    #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    resample=(;type=CV())
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.CV

dsc = setup_dataset(
    Xc, yc;(
    resample=(;type=Holdout()))
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.Holdout

dsc = setup_dataset(
    Xc, yc;
    resample=(;type=StratifiedCV())
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.StratifiedCV

dsc = setup_dataset(
    Xc, yc;
    resample=(;type=TimeSeriesCV())
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.TimeSeriesCV

dsc = setup_dataset(
    Xc, yc;
    resample=(;type=CV(nfolds=10, shuffle=true))
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}

# ---------------------------------------------------------------------------- #
#                              rng propagation                                 #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    resample=(type=CV(nfolds=10, shuffle=true), rng=Xoshiro(1))
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.mach.model.rng isa Xoshiro
@test dsc.pinfo.rng isa Xoshiro

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)

dsc = setup_dataset(
    Xc, yc;
    model=ModalDecisionTree(),
    resample=(type=CV(nfolds=5, shuffle=true), rng=Xoshiro(1)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test dsc.mach.model.model.rng isa Xoshiro
@test dsc.mach.model.tuning.rng isa Xoshiro
@test dsc.mach.model.resampling.rng isa Xoshiro

# ---------------------------------------------------------------------------- #
#                            validate modelsetup                               #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(;max_depth=5)
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.mach.model.max_depth == 5

@test_throws UndefVarError setup_dataset(
    Xc, yc;
    model=Invalid(;max_depth=5)
)

@test_throws MethodError setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(;invalid=5)
)

@test_throws MethodError setup_dataset(
    Xc, yc;
    resample=(train_ratio=0.5, invalid=maximum)
)

# ---------------------------------------------------------------------------- #
#                                    tuning                                    #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)

dsr = setup_dataset(
    Xr, yr;
    model=DecisionTreeRegressor(),
    resample=(;rng=Xoshiro(1234)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms)
)
@test dsr isa SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel}

range = (SX.range(:min_purity_increase, lower=0.001, upper=1.0, scale=:log),
     SX.range(:max_depth, lower=1, upper=10))

dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=(;rng=Xoshiro(1234)),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms)
)
@test dsc isa SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel}

selector = FeatureSelector()
range = MLJ.range(selector, :features, values = [[:sepal_width,], [:sepal_length, :sepal_width]])

dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=(;rng=Xoshiro(1234)),
    tuning=(;tuning=Grid(resolution=10),resampling=CV(nfolds=3),range,measure=rms)
)
@test dsc isa SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel}    

# # ---------------------------------------------------------------------------- #
# #                                dataset info                                  #
# # ---------------------------------------------------------------------------- #
# dsc = setup_dataset(Xc, yc)

# @test SX.get_treatment(dsc.info) == :aggregate
# @test SX.get_modalreduce(dsc.info) == mean
# @test SX.get_train_ratio(dsc.info) == 0.7
# @test SX.get_valid_ratio(dsc.info) == 0.0
# @test SX.get_rng(dsc.info) == TaskLocalRNG()
# @test SX.get_vnames(dsc.info) isa Vector{String}
# @test_nowarn sprint(show, dsc.info)

# output = sprint(show, dsc.info)
# @test occursin("DatasetInfo:", output)
# @test occursin("treatment:", output)
# @test occursin("aggregate", output)

# # ---------------------------------------------------------------------------- #
# #                                     tt                                       #
# # ---------------------------------------------------------------------------- #
# @test SX.get_train(dsc.tt[1]) isa Vector{Int64}
# @test isempty(SX.get_valid(dsc.tt[1]))
# @test SX.get_test(dsc.tt[1]) isa Vector{Int64}
# @test length(dsc.tt[1]) == 150
# @test_nowarn sprint(show, dsc.tt[1])

# # ---------------------------------------------------------------------------- #
# #                                   dataset                                    #
# # ---------------------------------------------------------------------------- #
# @test SX.get_X(dsc) isa Matrix
# @test SX.get_y(dsc) isa AbstractVector
# @test SX.get_tt(dsc) isa AbstractVector
# @test SX.get_info(dsc) isa SX.DatasetInfo
# @test_nowarn sprint(show, dsc)

# # ---------------------------------------------------------------------------- #
# #                         check vnames and modalreduce                         #
# # ---------------------------------------------------------------------------- #
# vnames=[:p1, :p2, :p3, :p4]
# dsc, _ = setup_dataset(
#     Xc, yc;
#     preprocess=(;vnames)
# )
# @test dsc.setup.preprocess.vnames == vnames

# _, dsmin = setup_dataset(Xts, yts; model=(;type=:modaldecisiontree), preprocess=(;modalreduce=minimum))
# _, dsmax = setup_dataset(Xts, yts; model=(;type=:modaldecisiontree), preprocess=(;modalreduce=maximum))
# @test all(dsmin.X .<= dsmax.X)

# # ---------------------------------------------------------------------------- #
# #                                  windowing                                   #
# # ---------------------------------------------------------------------------- #
# dsts, dts = setup_dataset(
#     Xts, yts;
#     win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1))
# )
# @test dsts isa SX.Modelset
# @test dts    isa SX.Dataset