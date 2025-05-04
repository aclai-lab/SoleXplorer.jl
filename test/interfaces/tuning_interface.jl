using Test
using SoleXplorer
using MLJ

@testset "TuningInterface" begin
    
    @testset "Type Instantiation" begin
        # Test TuningStrategy instantiation
        ts = SoleXplorer.TuningStrategy(grid, (resolution=5, shuffle=false))
        @test ts.type === grid
        @test ts.params.resolution == 5
        @test ts.params.shuffle == false
        
        # Test TuningParams instantiation
        dummy_range(model) = MLJ.range(model, :dummy, values=[1,2,3])
        tp = SoleXplorer.TuningParams(
            ts,
            (n=10, train_best=false),
            (dummy_range,)
        )
        @test tp.method === ts
        @test tp.params.n == 10
        @test tp.params.train_best == false
        @test length(tp.ranges) == 1
    end
    
    @testset "Available Tuning Methods" begin
        # Test that all expected methods are available
        @test grid                  in SoleXplorer.AVAIL_TUNING_METHODS
        @test randomsearch          in SoleXplorer.AVAIL_TUNING_METHODS
        @test latinhypercube        in SoleXplorer.AVAIL_TUNING_METHODS
        @test treeparzen            in SoleXplorer.AVAIL_TUNING_METHODS
        @test particleswarm         in SoleXplorer.AVAIL_TUNING_METHODS
        @test adaptiveparticleswarm in SoleXplorer.AVAIL_TUNING_METHODS
        
        # Test length
        @test length(SoleXplorer.AVAIL_TUNING_METHODS) == 6
    end
    
    @testset "Tuning Methods Parameters" begin
        # Test that all methods have parameters
        for method in SoleXplorer.AVAIL_TUNING_METHODS
            @test haskey(SoleXplorer.TUNING_METHODS_PARAMS, method)
            @test SoleXplorer.TUNING_METHODS_PARAMS[method] isa NamedTuple
        end
        
        # Test specific parameters
        @test SoleXplorer.TUNING_METHODS_PARAMS[grid].resolution == 10
        @test SoleXplorer.TUNING_METHODS_PARAMS[particleswarm].n_particles == 3
    end
    
    @testset "Tuning Parameters" begin
        # Test that classification and regression keys exist
        @test haskey(SoleXplorer.TUNING_PARAMS, :classification)
        @test haskey(SoleXplorer.TUNING_PARAMS, :regression)
        
        # Test specific parameters
        @test SoleXplorer.TUNING_PARAMS[:classification].n == 25
        @test SoleXplorer.TUNING_PARAMS[:regression].n == 25
        @test typeof(SoleXplorer.TUNING_PARAMS[:classification].measure) == typeof(MLJ.LogLoss())
        @test typeof(SoleXplorer.TUNING_PARAMS[:regression].measure) == typeof(MLJ.RootMeanSquaredError())
    end
    
    @testset "Range Function" begin
        # Test with values
        r1 = SoleXplorer.range(:max_depth, lower=2, upper=10)
        @test r1.field == :max_depth
        @test r1.lower == 2
        @test r1.upper == 10
        
        # Test with lower/upper
        r2 = SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        @test r2.field == :feature_importance
        @test r2.values == [:impurity, :split]
    end
end