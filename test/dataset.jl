using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SX.NatopsLoader()
Xts, yts = SX.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                        prepare dataset usage examples                        #
# ---------------------------------------------------------------------------- #
# basic setup
dsc = setup_dataset(Xc, yc)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}
dsr = setup_dataset(Xr, yr)
@test dsr isa SX.PropositionalDataSet{SX.DecisionTreeRegressor}

# model type specification
dsc = setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier()
)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Xc, yc;
    model=SX.RandomForestClassifier()
)
@test dsc isa SX.PropositionalDataSet{SX.RandomForestClassifier}

dsc = setup_dataset(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier()
)
@test dsc isa SX.PropositionalDataSet{SX.AdaBoostStumpClassifier}

dsr = setup_dataset(
    Xr, yr;
    model=SX.DecisionTreeRegressor()
)
@test dsr isa SX.PropositionalDataSet{SX.DecisionTreeRegressor}

dsr = setup_dataset(
    Xr, yr;
    model=SX.RandomForestRegressor()
)
@test dsr isa SX.PropositionalDataSet{SX.RandomForestRegressor}

dsts = setup_dataset(
    Xts, yts;
    model=SX.ModalDecisionTree()
)
@test dsts isa SX.ModalDataSet{SX.ModalDecisionTree}

dsts = setup_dataset(
    Xts, yts;
    model=SX.ModalRandomForest()
)
@test dsts isa SX.ModalDataSet{SX.ModalRandomForest}

dsts = setup_dataset(
    Xts, yts;
    model=SX.ModalAdaBoost()
)
@test dsts isa SX.ModalDataSet{SX.ModalAdaBoost}

dsc = setup_dataset(
    Xc, yc;
    model=SX.XGBoostClassifier()
)
@test dsc isa SX.PropositionalDataSet{SX.XGBoostClassifier}

dsr = setup_dataset(
    Xr, yr;
    model=SX.XGBoostRegressor()
)
@test dsr isa SX.PropositionalDataSet{SX.XGBoostRegressor}

# ---------------------------------------------------------------------------- #
#                                 code dataset                                 #
# ---------------------------------------------------------------------------- #
Xcd = DataFrame(
    numeric_int = [1, 2, 3, 4],
    numeric_float = [1.1, 2.2, 3.3, 4.4],
    categorical_string = ["A", "B", "A", "C"],
    boolean = [true, false, true, false]
)
ydc = ["class1", "class2", "class1", "class3"]

# apply encoding
coded_Xcd = code_dataset(Xcd)
coded_ydc = code_dataset(ydc)
coded_ds  = code_dataset(Xcd, ydc)

@test eltype(Xcd.categorical_string) <: Number
@test eltype(Xcd.boolean) <: Number
@test eltype(coded_ydc) <: Number

@test eltype(coded_ds[1].categorical_string) <: Number
@test eltype(coded_ds[1].boolean) <: Number
@test eltype(coded_ds[2]) <: Number

# test that encoding is consistent
@test Xcd.categorical_string[1] == Xcd.categorical_string[3]  # both "A"
@test coded_ydc[1] == coded_ydc[3]  # both "class1"
@test coded_ds[1].categorical_string[1] == coded_ds[1].categorical_string[3]  # both "A"
@test coded_ds[2][1] == coded_ds[2][3]  # both "class1"

# ---------------------------------------------------------------------------- #
#                covering various examples to complete codecov                 #
# ---------------------------------------------------------------------------- #
y_symbol = :petal_width
dsc = setup_dataset(Xc, y_symbol)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeRegressor}


# dataset is composed also of non numeric columns
Xnn = hcat(Xc, DataFrame(target = yc))
@test_nowarn SX.code_dataset(Xnn)

dsc = setup_dataset(
    Xts, yts;
    resampling=Holdout(fraction_train=0.5, shuffle=true),
    reducefunc=maximum
)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}

# ---------------------------------------------------------------------------- #
#                                 resamplig                                    #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    resampling=CV(),
)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.CV

dsc = setup_dataset(
    Xc, yc;
    resampling=Holdout(),
)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.Holdout

dsc = setup_dataset(
    Xc, yc;
    resampling=StratifiedCV(),
)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.StratifiedCV

dsc = setup_dataset(
    Xc, yc;
    resampling=TimeSeriesCV(),
)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.TimeSeriesCV

dsc = setup_dataset(
    Xc, yc;
    resampling=CV(nfolds=10, shuffle=true),
)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}

# ---------------------------------------------------------------------------- #
#                              seed propagation                                #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    resampling=CV(nfolds=10, shuffle=true),
    seed=1
)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}
@test dsc.mach.model.rng isa Xoshiro
@test dsc.pinfo.rng isa Xoshiro

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)

dsc = setup_dataset(
    Xc, yc;
    model=SX.ModalDecisionTree(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=SX.accuracy, repeats=2)
)
@test dsc.mach.model.model.rng isa Xoshiro
@test dsc.mach.model.tuning.rng isa Xoshiro
@test dsc.mach.model.resampling.rng isa Xoshiro

# ---------------------------------------------------------------------------- #
#                            validate modelsetup                               #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier(;max_depth=5)
)
@test dsc isa SX.PropositionalDataSet{SX.DecisionTreeClassifier}
@test dsc.mach.model.max_depth == 5

@test_throws UndefVarError setup_dataset(
    Xc, yc;
    model=Invalid(;max_depth=5)
)

@test_throws MethodError setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier(;invalid=5)
)

@test_throws MethodError setup_dataset(
    Xc, yc;
    resampling=Holdout(fraction_train=0.5, shuffle=true),
    invalid=maximum
)

# ---------------------------------------------------------------------------- #
#                                    tuning                                    #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsr = setup_dataset(
    Xr, yr;
    model=SX.DecisionTreeRegressor(),
    seed=1234,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=rms)
)
@test dsr isa SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel}
model = dsr.mach.model
@test model isa MLJ.MLJTuning.DeterministicTunedModel
@test model.tuning isa MLJ.MLJTuning.Grid

range = (SX.range(:min_purity_increase, lower=0.001, upper=1.0, scale=:log),
     SX.range(:max_depth, lower=1, upper=10))
dsc = setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    seed=1234,
    tuning=RandomTuning(range=range)
)
@test dsc isa SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel}
model = dsc.mach.model
@test model isa MLJ.MLJTuning.ProbabilisticTunedModel
@test model.tuning isa MLJ.MLJTuning.RandomSearch

selector = FeatureSelector()
range = MLJ.range(selector, :features, values = [[:sepal_width,], [:sepal_length, :sepal_width]])
dsc = setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    seed=1234,
    tuning=CubeTuning(resampling=CV(nfolds=3), range=range, measure=rms)
)
@test dsc isa SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel}
model = dsc.mach.model
@test model isa MLJ.MLJTuning.ProbabilisticTunedModel
@test model.tuning isa MLJ.MLJTuning.LatinHypercube

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsr = setup_dataset(
    Xr, yr;
    model=SX.DecisionTreeRegressor(),
    seed=1234,
    tuning=ParticleTuning(n_particles=3, resampling=CV(nfolds=3), range=range, measure=rms)
)
@test dsr isa SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel}
model = dsr.mach.model
@test model isa MLJ.MLJTuning.DeterministicTunedModel
@test model.tuning isa SX.MLJParticleSwarmOptimization.ParticleSwarm

range = (SX.range(:min_purity_increase, lower=0.001, upper=1.0, scale=:log),
     SX.range(:max_depth, lower=1, upper=10))
dsc = setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier(),
    seed=1234,
    tuning=AdaptiveTuning(range=range)
)
@test dsc isa SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel}
model = dsc.mach.model
@test model isa MLJ.MLJTuning.ProbabilisticTunedModel
@test model.tuning isa SX.MLJParticleSwarmOptimization.AdaptiveParticleSwarm

tuning=GridTuning(resolution=10, range=range)
@test propertynames(tuning) == (:strategy, :range, :resampling, :measure, :repeats)
@test SX.getproperty(tuning, :resampling) isa Holdout
@test get_strategy(tuning) isa MLJ.Grid

# ---------------------------------------------------------------------------- #
#                               various cases                                  #
# ---------------------------------------------------------------------------- #
y_invalid = fill(nothing, length(yc)) 
@test_throws ArgumentError setup_dataset(Xc, y_invalid)

@test SX.code_dataset(yc) isa Vector{Int64}
@test SX.code_dataset(Xc, yc) isa Tuple{DataFrame, Vector{Int64}}

dsc = setup_dataset(Xc, yc)
@test length(dsc) == length(dsc.pidxs)

@test SX.get_X(dsc, :train) isa Vector{<:AbstractDataFrame}
@test SX.get_y(dsc, :test) isa Vector{<:AbstractVector{<:SX.CLabel}}
@test SX.get_mach_model(dsc) isa SX.DecisionTreeClassifier

@test_nowarn dsc.pinfo
@test_nowarn dsc.pidxs
@test_nowarn length(dsc.pidxs)

@test length(dsc.pidxs) == length(dsc)

# ---------------------------------------------------------------------------- #
#                                  Base.show                                   #
# ---------------------------------------------------------------------------- #
@testset "Base.show tests for partition.jl" begin
    # Setup test data
    rng = Xoshiro(42)
    y = ["A", "B", "A", "B", "A", "B", "A", "B"]
    resampling = CV(nfolds=3)
    
    @testset "PartitionInfo show methods" begin
        pinfo = SX.PartitionInfo(resampling, 0.2, rng)
        
        # Test Base.show(io::IO, info::PartitionInfo)
        io = IOBuffer()
        show(io, pinfo)
        output = String(take!(io))
        
        @test occursin("PartitionInfo:", output)
        @test occursin("type:", output)
        @test occursin("valid_ratio:", output)
        @test occursin("rng:", output)
        @test occursin("0.2", output)
        
        # Test Base.show(io::IO, ::MIME"text/plain", info::PartitionInfo)
        io = IOBuffer()
        show(io, MIME("text/plain"), pinfo)
        plain_output = String(take!(io))
        
        @test plain_output == output  # Should be identical
    end
    
    @testset "PartitionIdxs show methods" begin
        # Create test PartitionIdxs
        train_idxs = [1, 2, 3, 4]
        valid_idxs = [5, 6]
        test_idxs = [7, 8]
        pidx = SX.PartitionIdxs(train_idxs, valid_idxs, test_idxs)
        
        # Test Base.show(io::IO, pidx::PartitionIdxs{T})
        io = IOBuffer()
        show(io, pidx)
        output = String(take!(io))
        
        @test occursin("PartitionIdxs{Int", output)
        @test occursin("Total samples: 8", output)
        @test occursin("Train: 4", output)
        @test occursin("Valid: 2", output)
        @test occursin("Test: 2", output)
        
        # Test Base.show(io::IO, ::MIME"text/plain", pidx::PartitionIdxs{T})
        io = IOBuffer()
        show(io, MIME("text/plain"), pidx)
        plain_output = String(take!(io))
        
        @test plain_output == output  # Should be identical
        
        # Test with empty valid set
        pidx_no_valid = SX.PartitionIdxs([1, 2, 3, 4, 5], Int[], [6, 7, 8])
        io = IOBuffer()
        show(io, pidx_no_valid)
        output_no_valid = String(take!(io))
        
        @test occursin("Total samples: 8", output_no_valid)
        @test occursin("Train: 5", output_no_valid)
        @test occursin("Valid: 0", output_no_valid)
        @test occursin("Test: 3", output_no_valid)
    end
    
    @testset "Integration test with partition function" begin
        # Test show methods with actual partition results
        pidxs, pinfo = SX.partition(y; resampling, valid_ratio=0.2, rng=rng)
        
        # Test PartitionInfo show
        io = IOBuffer()
        show(io, pinfo)
        pinfo_output = String(take!(io))
        @test occursin("PartitionInfo:", pinfo_output)
        @test occursin("CV", pinfo_output)
        
        # Test PartitionIdxs show for first fold
        io = IOBuffer()
        show(io, pidxs[1])
        pidx_output = String(take!(io))
        @test occursin("PartitionIdxs{Int", pidx_output)
        @test occursin("Total samples:", pidx_output)
        @test occursin("Train:", pidx_output)
        @test occursin("Valid:", pidx_output)
        @test occursin("Test:", pidx_output)
    end
end

@testset "Tuning show methods" begin
    # Create a test tuning configuration
    tuning = GridTuning(
        range=(:max_depth, 1:10),
        measure=SX.accuracy,
        repeats=2
    )
    
    # Test text/plain show method
    io = IOBuffer()
    show(io, MIME"text/plain"(), tuning)
    output = String(take!(io))
    
    @test contains(output, "Tuning{")
    @test contains(output, "strategy:")
    @test contains(output, "range:")
    @test contains(output, "resampling:")
    @test contains(output, "measure:")
    @test contains(output, "repeats:")
    
    # Test compact show method
    io = IOBuffer()
    show(io, tuning)
    compact_output = String(take!(io))
    
    @test contains(compact_output, "Tuning{")
    @test contains(compact_output, "repeats=2")
    
    # Test that it doesn't error when printed
    @test_nowarn println(tuning)
    @test_nowarn display(tuning)
end