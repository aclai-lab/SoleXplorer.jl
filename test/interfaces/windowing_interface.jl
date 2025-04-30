using Test
using SoleXplorer

@testset "WinParams Struct" begin
    # Create a WinParams instance for movingwindow
    params = SoleXplorer.WinParams(
        movingwindow, 
        (window_size = 2048, window_step = 1024)
    )
    
    # Test structure
    @test params.type == movingwindow
    @test params.params isa NamedTuple
    @test params.params.window_size == 2048
    @test params.params.window_step == 1024
    
    # Create a WinParams instance for splitwindow
    params2 = SoleXplorer.WinParams(
        splitwindow, 
        (nwindows = 10,)
    )
    
    # Test structure
    @test params2.type == splitwindow
    @test params2.params isa NamedTuple
    @test params2.params.nwindows == 10
end

@testset "Windowing Constants" begin
    # Test that window types exist
    @test :movingwindow   in propertynames(SoleXplorer)
    @test :wholewindow    in propertynames(SoleXplorer)
    @test :splitwindow    in propertynames(SoleXplorer)
    @test :adaptivewindow in propertynames(SoleXplorer)
    
    # Test available windows constants
    @test SoleXplorer.AVAIL_WINS isa Tuple
    @test length(SoleXplorer.AVAIL_WINS) == 4
    @test all(w -> w isa Function, SoleXplorer.AVAIL_WINS)
    
    # Test feature extraction available windows
    @test SoleXplorer.FE_AVAIL_WINS isa Tuple
    @test length(SoleXplorer.FE_AVAIL_WINS) == 3
    @test all(w -> w isa Function, SoleXplorer.FE_AVAIL_WINS)
    
    # Test window parameters
    @test SoleXplorer.WIN_PARAMS isa Dict
    @test haskey(SoleXplorer.WIN_PARAMS, movingwindow)
    @test haskey(SoleXplorer.WIN_PARAMS, wholewindow)
    @test haskey(SoleXplorer.WIN_PARAMS, splitwindow)
    @test haskey(SoleXplorer.WIN_PARAMS, adaptivewindow)
    
    # Test specific window parameters
    @test SoleXplorer.WIN_PARAMS[movingwindow].window_size        == 1024
    @test SoleXplorer.WIN_PARAMS[movingwindow].window_step        == 512
    @test SoleXplorer.WIN_PARAMS[wholewindow]                     isa NamedTuple
    @test isempty(SoleXplorer.WIN_PARAMS[wholewindow])
    @test SoleXplorer.WIN_PARAMS[splitwindow].nwindows            == 5
    @test SoleXplorer.WIN_PARAMS[adaptivewindow].nwindows         == 5
    @test SoleXplorer.WIN_PARAMS[adaptivewindow].relative_overlap == 0.1
end

@testset "InfoFeat Struct" begin
    # Test constructor with Symbol variable name
    info_feat1 = SoleXplorer.InfoFeat(1, :temperature, :mean, 2)
    @test info_feat1.id   == 1
    @test info_feat1.var  == :temperature
    @test info_feat1.feat == :mean
    @test info_feat1.nwin == 2
    
    # Test constructor with String variable name
    info_feat2 = SoleXplorer.InfoFeat(42, "heart_rate", :periodicity, 3)
    @test info_feat2.id   == 42
    @test info_feat2.var  == "heart_rate"
    @test info_feat2.feat == :periodicity
    @test info_feat2.nwin == 3
    
    # Test accessor functions
    @test SoleXplorer.getproperty(info_feat1, :var) == :temperature
    @test SoleXplorer.propertynames(info_feat1)     == (:id, :feat, :var, :nwin)
    @test SoleXplorer.feature_id(info_feat1)        == 1
    @test SoleXplorer.variable_name(info_feat1)     == :temperature
    @test SoleXplorer.feature_type(info_feat1)      == :mean
    @test SoleXplorer.window_number(info_feat1)     == 2
    
    # Test argument validation
    @test_throws ArgumentError SoleXplorer.InfoFeat(5, :pressure, :std, 0)
    @test_throws ArgumentError SoleXplorer.InfoFeat(6, :pressure, :std, -1)
end
