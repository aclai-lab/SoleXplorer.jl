using Test
using SoleXplorer
const SX = SoleXplorer

using SoleData

using MLJ
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SoleData.Artifacts.NatopsLoader()
Xts, yts = SoleData.Artifacts.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                                     models                                   #
# ---------------------------------------------------------------------------- #
solemc = train_test(
    Xc, yc;
    model=SX.DecisionTreeClassifier()
)
@test solemc isa SX.SoleModel{SX.DataSet{SX.DecisionTreeClassifier, Int}}

solemc = train_test(
    Xc, yc;
    model=SX.RandomForestClassifier()
)
@test solemc isa SX.SoleModel{SX.DataSet{SX.RandomForestClassifier, Int}}

solemc = train_test(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier()
)
@test solemc isa SX.SoleModel{SX.DataSet{SX.AdaBoostStumpClassifier, Int}}

solemr = train_test(
    Xr, yr;
    model=SX.DecisionTreeRegressor()
)
@test solemr isa SX.SoleModel{SX.DataSet{SX.DecisionTreeRegressor, Int}}

solemr = train_test(
    Xr, yr;
    model=SX.RandomForestRegressor()
)
@test solemr isa SX.SoleModel{SX.DataSet{SX.RandomForestRegressor, Int}}

solemts = train_test(
    Xts, yts;
    model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(
        reducefunc=mean,
        win=(splitwindow(nwindows=5),)
    )
)
@test solemts isa SX.SoleModel{SX.DataSet{SX.ModalDecisionTree, Int}}

solemts = train_test(
    Xts, yts;
    model=SX.ModalRandomForest(),
    aggrfunc=reducesize(
        reducefunc=mean,
        win=(splitwindow(nwindows=5),)
    )    
)
@test solemts isa SX.SoleModel{SX.DataSet{SX.ModalRandomForest, Int}}

solemts = train_test(
    Xts, yts;
    model=SX.ModalAdaBoost(),
    aggrfunc=reducesize(
        reducefunc=mean,
        win=(splitwindow(nwindows=5),)
    ),
    float_type=Float64
)
@test solemts isa SX.SoleModel{SX.DataSet{SX.ModalAdaBoost, Int}}

solemc = train_test(
    Xc, yc;
    model=SX.XGBoostClassifier()
)
@test solemc isa SX.SoleModel{SX.DataSet{SX.XGBoostClassifier, Int}}

solemr = train_test(
    Xr, yr;
    model=SX.XGBoostRegressor()
)
@test solemr isa SX.SoleModel{SX.DataSet{SX.XGBoostRegressor, Int}}

# ---------------------------------------------------------------------------- #
#                                    various                                   #
# ---------------------------------------------------------------------------- #
@testset "Base.show tests for train_test.jl" begin
    # Create a dataset and train models
    ds = setup_dataset(
        Xc, yc;
        model=SX.DecisionTreeClassifier(),
        resampling=CV(nfolds=3, shuffle=true),
        rng=42
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
        Xc, yc;
        model = SX.DecisionTreeClassifier(),
        resampling = CV(nfolds=5),
        rng=42
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
