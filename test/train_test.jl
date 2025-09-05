using Test
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

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
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{DecisionTreeClassifier}}
solemr = train_test(Xr, yr)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{DecisionTreeRegressor}}

datac  = setup_dataset(Xc, yc)
solemc = train_test(datac)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{DecisionTreeClassifier}}
datar  = setup_dataset(Xr, yr)
solemr = train_test(datar)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{DecisionTreeRegressor}}

# ---------------------------------------------------------------------------- #
#                                     models                                   #
# ---------------------------------------------------------------------------- #
solemc = train_test(
    Xc, yc;
    model=DecisionTreeClassifier()
)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{DecisionTreeClassifier}}

solemc = train_test(
    Xc, yc;
    model=RandomForestClassifier()
)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{RandomForestClassifier}}

solemc = train_test(
    Xc, yc;
    model=AdaBoostStumpClassifier()
)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{AdaBoostStumpClassifier}}

solemr = train_test(
    Xr, yr;
    model=DecisionTreeRegressor()
)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{DecisionTreeRegressor}}

solemr = train_test(
    Xr, yr;
    model=RandomForestRegressor()
)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{RandomForestRegressor}}

solemts = train_test(
    Xts, yts;
    model=ModalDecisionTree()
)
@test solemts isa SX.SoleModel{SX.ModalDataSet{ModalDecisionTree}}

solemts = train_test(
    Xts, yts;
    model=ModalRandomForest()
)
@test solemts isa SX.SoleModel{SX.ModalDataSet{ModalRandomForest}}

solemts = train_test(
    Xts, yts;
    model=ModalAdaBoost()
)
@test solemts isa SX.SoleModel{SX.ModalDataSet{ModalAdaBoost}}

solemc = train_test(
    Xc, yc;
    model=XGBoostClassifier()
)
@test solemc isa SX.SoleModel{SX.PropositionalDataSet{XGBoostClassifier}}

solemr = train_test(
    Xr, yr;
    model=XGBoostRegressor()
)
@test solemr isa SX.SoleModel{SX.PropositionalDataSet{XGBoostRegressor}}

# ---------------------------------------------------------------------------- #
#                                     tuning                                   #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemc = train_test(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemc isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:DecisionTreeClassifier}}}

range = (
    SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log),
    SX.range(:n_trees; lower=10, unit=20, upper=90)
)
solemc = train_test(
    Xc, yc;
    model=RandomForestClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemc isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:RandomForestClassifier}}}

range = SX.range(:n_iter; lower=10, unit=10, upper=100)
solemc = train_test(
    Xc, yc;
    model=AdaBoostStumpClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemc isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:AdaBoostStumpClassifier}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemr = train_test(
    Xr, yr;
    model=DecisionTreeRegressor(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)
)
@test solemr isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:DecisionTreeRegressor}}}

range = (
    SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log),
    SX.range(:n_trees; lower=10, unit=20, upper=90)
)
solemr = train_test(
    Xr, yr;
    model=RandomForestRegressor(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)
)
@test solemr isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:RandomForestRegressor}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemts = train_test(
    Xts, yts;
    model=ModalDecisionTree(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemts isa SX.SoleModel{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:ModalDecisionTree}}}

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
solemts = train_test(
    Xts, yts;
    model=ModalRandomForest(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemts isa SX.SoleModel{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:ModalRandomForest}}}

range = SX.range(:n_iter; lower=2, unit=10, upper=10)
solemts = train_test(
    Xts, yts;
    model=ModalAdaBoost(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemts isa SX.SoleModel{<:SX.ModalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:ModalAdaBoost}}}

range = SX.range(:num_round; lower=10, unit=10, upper=100)
solemc = train_test(
    Xc, yc;
    model=XGBoostClassifier(
        early_stopping_rounds=20,
    ),
    resample=CV(nfolds=5, shuffle=true),
    valid_ratio=0.2,
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)
)
@test solemc isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:XGBoostClassifier}}}

range = SX.range(:num_round; lower=10, unit=10, upper=100)
solemr = train_test(
    Xr, yr;
    model=XGBoostRegressor(
        early_stopping_rounds=20,
    ),
    resample=CV(nfolds=5, shuffle=true),
    valid_ratio=0.2,
    rng=Xoshiro(1),
    tuning=(;tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=rms, repeats=2)
)
@test solemr isa SX.SoleModel{<:SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:XGBoostRegressor}}}

# ---------------------------------------------------------------------------- #
#                                    various                                   #
# ---------------------------------------------------------------------------- #
@testset "Base.show tests for train_test.jl" begin
    rng = Xoshiro(42)
    
    # Create a dataset and train models
    ds = setup_dataset(
        Xc, yc,
        model = DecisionTreeClassifier(),
        resample = CV(nfolds=3, shuffle=true),
        train_ratio = 0.7,
        rng = rng
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
        model = DecisionTreeClassifier(),
        resample = CV(nfolds=5),
        train_ratio = 0.8,
        rng = rng
    )
    
    solem_5fold = train_test(ds_5fold)
    
    io = IOBuffer()
    show(io, solem_5fold)
    output_5fold = String(take!(io))
    
    @test occursin("Number of models: 5", output_5fold)

    
    # @testset "SoleModel show with different dataset types" begin
    #     # Test with regression dataset
    #     X_reg = DataFrame(
    #         x1 = randn(rng, 10),
    #         x2 = randn(rng, 10)
    #     )
    #     y_reg = randn(rng, 10)
        
    #     ds_reg = setup_dataset(
    #         X_reg, y_reg,
    #         model = DecisionTreeRegressor(),
    #         resample = CV(nfolds=2),
    #         train_ratio = 0.7,
    #         rng = rng
    #     )
        
    #     solem_reg = train_test(ds_reg)
        
    #     io = IOBuffer()
    #     show(io, solem_reg)
    #     output_reg = String(take!(io))
        
    #     @test occursin("SoleModel{", output_reg)
    #     @test occursin("Number of models: 2", output_reg)
    #     @test occursin("DataSet", output_reg)
    # end
end
