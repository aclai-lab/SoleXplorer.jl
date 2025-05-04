using Test
using SoleXplorer
using Random

@testset "Resample Interface" begin
    
    @testset "Resample Struct" begin
        # Test Resample constructor
        cv_params = (nfolds=5, shuffle=true, rng=TaskLocalRNG())
        resample = SoleXplorer.Resample(CV, cv_params)
        
        @test resample.type                   == CV
        @test resample.params                 == cv_params
        @test resample.params.nfolds          == 5
        @test resample.params.shuffle         == true
        
        # Test with different resampling strategy
        holdout_params = (fraction_train=0.8, shuffle=false, rng=TaskLocalRNG())
        resample2 = SoleXplorer.Resample(Holdout, holdout_params)
        
        @test resample2.type                  == Holdout
        @test resample2.params                == holdout_params
        @test resample2.params.fraction_train == 0.8
        @test resample2.params.shuffle        == false
    end
    
    @testset "Available Resamples" begin
        # Test that all expected resampling strategies are available
        @test CV           in SoleXplorer.AVAIL_RESAMPLES
        @test Holdout      in SoleXplorer.AVAIL_RESAMPLES
        @test StratifiedCV in SoleXplorer.AVAIL_RESAMPLES
        @test TimeSeriesCV in SoleXplorer.AVAIL_RESAMPLES
        
        # Test that the constant has exactly these four strategies
        @test length(SoleXplorer.AVAIL_RESAMPLES) == 4
        @test Set(SoleXplorer.AVAIL_RESAMPLES)    == Set([CV, Holdout, StratifiedCV, TimeSeriesCV])
    end
    
    @testset "Resample Parameters" begin
        # Test that default parameters are defined for each resampling strategy
        @test haskey(SoleXplorer.RESAMPLE_PARAMS, CV)
        @test haskey(SoleXplorer.RESAMPLE_PARAMS, Holdout)
        @test haskey(SoleXplorer.RESAMPLE_PARAMS, StratifiedCV)
        @test haskey(SoleXplorer.RESAMPLE_PARAMS, TimeSeriesCV)
        
        # Test specific parameter values
        @test SoleXplorer.RESAMPLE_PARAMS[CV].nfolds == 6
        @test SoleXplorer.RESAMPLE_PARAMS[CV].shuffle == true
        
        @test SoleXplorer.RESAMPLE_PARAMS[Holdout].fraction_train == 0.7
        @test SoleXplorer.RESAMPLE_PARAMS[Holdout].shuffle == true
        
        @test SoleXplorer.RESAMPLE_PARAMS[StratifiedCV].nfolds == 6
        @test SoleXplorer.RESAMPLE_PARAMS[StratifiedCV].shuffle == true
        
        @test SoleXplorer.RESAMPLE_PARAMS[TimeSeriesCV].nfolds == 4
        @test !haskey(SoleXplorer.RESAMPLE_PARAMS[TimeSeriesCV], :shuffle)  # TimeSeriesCV doesn't have shuffle param
    end
    
    @testset "Creating Resamples from Default Parameters" begin
        # Test creating Resample instances using the default parameters
        for resampler in SoleXplorer.AVAIL_RESAMPLES
            default_params = SoleXplorer.RESAMPLE_PARAMS[resampler]
            resample = SoleXplorer.Resample(resampler, default_params)
            
            @test resample.type == resampler
            @test resample.params == default_params
        end
    end
end