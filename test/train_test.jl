using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = NatopsLoader()
Xts, yts = SX.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                        train and test usage examples                         #
# ---------------------------------------------------------------------------- #
# basic setup
solemc = train_test(Xc, yc)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{SX.DecisionTreeClassifier}}
solemr = train_test(Xr, yr)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{SX.DecisionTreeRegressor}}

datac  = setup_dataset(Xc, yc)
solemc = train_test(datac)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{SX.DecisionTreeClassifier}}
datar  = setup_dataset(Xr, yr)
solemr = train_test(datar)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{SX.DecisionTreeRegressor}}

# ---------------------------------------------------------------------------- #
#                                     models                                   #
# ---------------------------------------------------------------------------- #
solemc = train_test(
    Xc, yc;
    model=SX.DecisionTreeClassifier()
)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{SX.DecisionTreeClassifier}}

solemc = train_test(
    Xc, yc;
    model=SX.RandomForestClassifier()
)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{SX.RandomForestClassifier}}

solemc = train_test(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier()
)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{SX.AdaBoostStumpClassifier}}

solemr = train_test(
    Xr, yr;
    model=SX.DecisionTreeRegressor()
)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{SX.DecisionTreeRegressor}}

solemr = train_test(
    Xr, yr;
    model=SX.RandomForestRegressor()
)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{SX.RandomForestRegressor}}

solemts = train_test(
    Xts, yts;
    model=SX.ModalDecisionTree()
)
@test solemts isa SX.SoleModel{SX.ModalDataSet{SX.ModalDecisionTree}}

solemts = train_test(
    Xts, yts;
    model=SX.ModalRandomForest()
)
@test solemts isa SX.SoleModel{SX.ModalDataSet{SX.ModalRandomForest}}

solemts = train_test(
    Xts, yts;
    model=SX.ModalAdaBoost()
)
@test solemts isa SX.SoleModel{SX.ModalDataSet{SX.ModalAdaBoost}}

solemc = train_test(
    Xc, yc;
    model=SX.XGBoostClassifier()
)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{SX.XGBoostClassifier}}

solemr = train_test(
    Xr, yr;
    model=SX.XGBoostRegressor()
)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{SX.XGBoostRegressor}}

# ---------------------------------------------------------------------------- #
#                                     tuning                                   #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemc = train_test(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2)
)
@test solemc isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:SX.DecisionTreeClassifier}}}

range = (
    SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log),
    SX.range(:n_trees; lower=10, unit=20, upper=90)
)
solemc = train_test(
    Xc, yc;
    model=SX.RandomForestClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2)
)
@test solemc isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:SX.RandomForestClassifier}}}

range = SX.range(:n_iter; lower=10, unit=10, upper=100)
solemc = train_test(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2)
)
@test solemc isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:SX.AdaBoostStumpClassifier}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemr = train_test(
    Xr, yr;
    model=SX.DecisionTreeRegressor(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=rms, repeats=2)
)
@test solemr isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:SX.DecisionTreeRegressor}}}

range = (
    SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log),
    SX.range(:n_trees; lower=10, unit=20, upper=90)
)
solemr = train_test(
    Xr, yr;
    model=SX.RandomForestRegressor(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=rms, repeats=2)
)
@test solemr isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:SX.RandomForestRegressor}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemts = train_test(
    Xts, yts;
    model=SX.ModalDecisionTree(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2)
)
@test solemts isa SX.SoleModel{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:SX.ModalDecisionTree}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemts = train_test(
    Xts, yts;
    model=SX.ModalRandomForest(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2)
)
@test solemts isa SX.SoleModel{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:SX.ModalRandomForest}}}

range = SX.range(:n_iter; lower=2, unit=10, upper=10)
solemts = train_test(
    Xts, yts;
    model=SX.ModalAdaBoost(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2)
)
@test solemts isa SX.SoleModel{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:SX.ModalAdaBoost}}}

range = SX.range(:num_round; lower=10, unit=10, upper=100)
solemc = train_test(
    Xc, yc;
    model=SX.XGBoostClassifier(
        early_stopping_rounds=20,
    ),
    resampling=CV(nfolds=5, shuffle=true),
    valid_ratio=0.2,
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2)
)
@test solemc isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:SX.XGBoostClassifier}}}

range = SX.range(:num_round; lower=10, unit=10, upper=100)
solemr = train_test(
    Xr, yr;
    model=SX.XGBoostRegressor(
        early_stopping_rounds=20,
    ),
    resampling=CV(nfolds=5, shuffle=true),
    valid_ratio=0.2,
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=rms, repeats=2)
)
@test solemr isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:SX.XGBoostRegressor}}}

# ---------------------------------------------------------------------------- #
#                                    various                                   #
# ---------------------------------------------------------------------------- #
@testset "Base.show tests for train_test.jl" begin
    # Create a dataset and train models
    ds = setup_dataset(
        Xc, yc,
        model=SX.DecisionTreeClassifier(),
        resampling=CV(nfolds=3, shuffle=true),
        seed=42
    )
    
    # Create SoleModel with trained models
    solem = train_test(ds)
    
    # Test Base.show(io::IO, solem::SoleModel{D})
    io = IOBuffer()
    show(io, solem)
    output = String(take!(io))
    
    @test occursin("SoleModel{", output)
    @test occursin("Number of models: 3", output)  # 3 folds
    @test occursin("DataSet", output)  # Should show dataset type
    
    # Test Base.show(io::IO, ::MIME"text/plain", solem::SoleModel{D})
    io = IOBuffer()
    show(io, MIME("text/plain"), solem)
    plain_output = String(take!(io))
    
    @test plain_output == output  # Should be identical
    
    # Test with different number of folds
    ds_5fold = setup_dataset(
        Xc, yc,
        model = SX.DecisionTreeClassifier(),
        resampling = CV(nfolds=5),
        seed=42
    )
    
    solem_5fold = train_test(ds_5fold)
    
    io = IOBuffer()
    show(io, solem_5fold)
    output_5fold = String(take!(io))
    
    @test occursin("Number of models: 5", output_5fold)
end

model = SX.XGBoostClassifier()
@test SX.has_xgboost_model(model) == true
model = SX.RandomForestClassifier()
@test SX.has_xgboost_model(model) == false

dsc = setup_dataset(Xc, yc)
@test SX.is_tuned_model(dsc.mach.model) == false
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsc = setup_dataset(
    Xc, yc;
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2)
)
@test SX.is_tuned_model(dsc.mach.model) == true
