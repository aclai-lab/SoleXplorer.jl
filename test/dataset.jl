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
    train_ratio=0.5,
    modalreduce=maximum
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}

# ---------------------------------------------------------------------------- #
#                                 resamplig                                    #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    resample=CV(),
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.CV

dsc = setup_dataset(
    Xc, yc;
    resample=Holdout(),
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.Holdout

dsc = setup_dataset(
    Xc, yc;
    resample=StratifiedCV(),
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.StratifiedCV

dsc = setup_dataset(
    Xc, yc;
    resample=TimeSeriesCV(),
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.TimeSeriesCV

dsc = setup_dataset(
    Xc, yc;
    resample=CV(nfolds=10, shuffle=true),
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}

# ---------------------------------------------------------------------------- #
#                              rng propagation                                 #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    resample=CV(nfolds=10, shuffle=true),
    rng=Xoshiro(1)
)
@test dsc isa SX.PropositionalDataSet{DecisionTreeClassifier}
@test dsc.mach.model.rng isa Xoshiro
@test dsc.pinfo.rng isa Xoshiro

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)

dsc = setup_dataset(
    Xc, yc;
    model=ModalDecisionTree(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=accuracy, repeats=2)
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
    train_ratio=0.5,
    invalid=maximum
)

# ---------------------------------------------------------------------------- #
#                                    tuning                                    #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsr = setup_dataset(
    Xr, yr;
    model=DecisionTreeRegressor(),
    rng=Xoshiro(1234),
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=rms)
)
@test dsr isa SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel}

range = (SX.range(:min_purity_increase, lower=0.001, upper=1.0, scale=:log),
     SX.range(:max_depth, lower=1, upper=10))
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    rng=Xoshiro(1234),
    tuning=RandomTuning(range=range)
)
@test dsc isa SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel}

selector = FeatureSelector()
range = MLJ.range(selector, :features, values = [[:sepal_width,], [:sepal_length, :sepal_width]])
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    rng=Xoshiro(1234),
    tuning=CubeTuning(resampling=CV(nfolds=3), range=range, measure=rms)
)
@test dsc isa SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel}  

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsr = setup_dataset(
    Xr, yr;
    model=DecisionTreeRegressor(),
    rng=Xoshiro(1234),
    tuning=ParticleTuning(n_particles=3, resampling=CV(nfolds=3), range=range, measure=rms)
)
@test dsr isa SX.PropositionalDataSet{<:MLJ.MLJTuning.DeterministicTunedModel}

range = (SX.range(:min_purity_increase, lower=0.001, upper=1.0, scale=:log),
     SX.range(:max_depth, lower=1, upper=10))
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    rng=Xoshiro(1234),
    tuning=AdaptiveTuning(range=range)
)
@test dsc isa SX.PropositionalDataSet{<:MLJ.MLJTuning.ProbabilisticTunedModel}

# ---------------------------------------------------------------------------- #
#                               various cases                                  #
# ---------------------------------------------------------------------------- #
y_invalid = fill(nothing, length(yc)) 
@test_throws ArgumentError setup_dataset(Xc, y_invalid)

@test SX.code_dataset!(yc) isa Vector{Int64}
@test SX.code_dataset!(Xc, yc) isa Tuple{DataFrame, Vector{Int64}}

dsc = setup_dataset(Xc, yc)
@test length(dsc) == length(dsc.pidxs)

@test SX.get_y_test(dsc) isa Vector{<:AbstractVector{<:SX.CLabel}}
@test SX.get_mach_model(dsc) isa DecisionTreeClassifier

@test_nowarn dsc.pinfo
@test_nowarn dsc.pidxs

@test length(dsc.pidxs) == length(dsc)

# ---------------------------------------------------------------------------- #
#                                  Base.show                                   #
# ---------------------------------------------------------------------------- #
@testset "Base.show tests for partition.jl" begin
    
    # Setup test data
    rng = Xoshiro(42)
    y = ["A", "B", "A", "B", "A", "B", "A", "B"]
    resample = CV(nfolds=3)
    
    @testset "PartitionInfo show methods" begin
        pinfo = SX.PartitionInfo(resample, 0.7, 0.2, rng)
        
        # Test Base.show(io::IO, info::PartitionInfo)
        io = IOBuffer()
        show(io, pinfo)
        output = String(take!(io))
        
        @test occursin("PartitionInfo:", output)
        @test occursin("type:", output)
        @test occursin("train_ratio:", output)
        @test occursin("valid_ratio:", output)
        @test occursin("rng:", output)
        @test occursin("0.7", output)
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
        pidxs, pinfo = SX.partition(y; resample=resample, train_ratio=0.7, valid_ratio=0.2, rng=rng)
        
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

using Test
using DataFrames

@testset "Base.show tests for treatment.jl" begin
    @testset "TreatmentInfo show method" begin
        # Create test TreatmentInfo
        features = (maximum, minimum, mean)
        win = AdaptiveWindow(nwindows=3, relative_overlap=0.1)
        treat = :aggregate
        modalreduce = mean
        
        tinfo = SX.TreatmentInfo(features, win, treat, modalreduce)
        
        # Test Base.show(io::IO, info::TreatmentInfo)
        io = IOBuffer()
        show(io, tinfo)
        output = String(take!(io))
        
        @test occursin("TreatmentInfo:", output)
        @test occursin("features:", output)
        @test occursin("winparams:", output)
        @test occursin("treatment:", output)
        @test occursin("modalreduce:", output)
        @test occursin("aggregate", output)
        @test occursin("mean", output)
        
        # Check that all fields are displayed with proper formatting
        lines = split(output, '\n')
        @test length(lines) >= 5  # Header + 4 fields + empty line at end
        @test occursin("features:", lines[2])
        @test occursin("winparams:", lines[3])
        @test occursin("treatment:", lines[4])
        @test occursin("modalreduce:", lines[5])
    end
    
    @testset "AggregationInfo show method" begin
        # Create test AggregationInfo
        features = (sum, std)
        win = SplitWindow(nwindows=5)
        
        ainfo = SX.AggregationInfo(features, win)
        
        # Test Base.show(io::IO, info::AggregationInfo)
        io = IOBuffer()
        show(io, ainfo)
        output = String(take!(io))
        
        @test occursin("AggregationInfo:", output)
        @test occursin("features:", output)
        @test occursin("winparams:", output)
        
        # Check that all fields are displayed with proper formatting
        lines = split(output, '\n')
        @test length(lines) >= 3  # Header + 2 fields + empty line at end
        @test occursin("features:", lines[2])
        @test occursin("winparams:", lines[3])
    end
end

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)

@btime setup_dataset(
    Xc, yc;
    model=ModalDecisionTree(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=accuracy, repeats=2)
);