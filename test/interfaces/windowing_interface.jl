using Test
using SymbolicAnalysis

@testset "WinParams Struct" begin
    # Create a WinParams instance for movingwindow
    params = SymbolicAnalysis.WinParams(
        movingwindow, 
        (window_size = 2048, window_step = 1024)
    )
    
    # Test structure
    @test params.type == movingwindow
    @test params.params isa NamedTuple
    @test params.params.window_size == 2048
    @test params.params.window_step == 1024
    
    # Create a WinParams instance for splitwindow
    params2 = SymbolicAnalysis.WinParams(
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
    @test :movingwindow   in propertynames(SymbolicAnalysis)
    @test :wholewindow    in propertynames(SymbolicAnalysis)
    @test :splitwindow    in propertynames(SymbolicAnalysis)
    @test :adaptivewindow in propertynames(SymbolicAnalysis)
    
    # Test available windows constants
    @test SymbolicAnalysis.AVAIL_WINS isa Tuple
    @test length(SymbolicAnalysis.AVAIL_WINS) == 4
    @test all(w -> w isa Function, SymbolicAnalysis.AVAIL_WINS)
    
    # Test feature extraction available windows
    @test SymbolicAnalysis.FE_AVAIL_WINS isa Tuple
    @test length(SymbolicAnalysis.FE_AVAIL_WINS) == 3
    @test all(w -> w isa Function, SymbolicAnalysis.FE_AVAIL_WINS)
    
    # Test window parameters
    @test SymbolicAnalysis.WIN_PARAMS isa Dict
    @test haskey(SymbolicAnalysis.WIN_PARAMS, movingwindow)
    @test haskey(SymbolicAnalysis.WIN_PARAMS, wholewindow)
    @test haskey(SymbolicAnalysis.WIN_PARAMS, splitwindow)
    @test haskey(SymbolicAnalysis.WIN_PARAMS, adaptivewindow)
    
    # Test specific window parameters
    @test SymbolicAnalysis.WIN_PARAMS[movingwindow].window_size        == 1024
    @test SymbolicAnalysis.WIN_PARAMS[movingwindow].window_step        == 512
    @test SymbolicAnalysis.WIN_PARAMS[wholewindow]                     isa NamedTuple
    @test isempty(SymbolicAnalysis.WIN_PARAMS[wholewindow])
    @test SymbolicAnalysis.WIN_PARAMS[splitwindow].nwindows            == 5
    @test SymbolicAnalysis.WIN_PARAMS[adaptivewindow].nwindows         == 5
    @test SymbolicAnalysis.WIN_PARAMS[adaptivewindow].relative_overlap == 0.1
end

@testset "InfoFeat Struct" begin
    # Test constructor with Symbol variable name
    info_feat1 = SymbolicAnalysis.InfoFeat(1, :temperature, :mean, 2)
    @test info_feat1.id   == 1
    @test info_feat1.var  == :temperature
    @test info_feat1.feat == :mean
    @test info_feat1.nwin == 2
    
    # Test constructor with String variable name
    info_feat2 = SymbolicAnalysis.InfoFeat(42, "heart_rate", :periodicity, 3)
    @test info_feat2.id   == 42
    @test info_feat2.var  == "heart_rate"
    @test info_feat2.feat == :periodicity
    @test info_feat2.nwin == 3
    
    # Test accessor functions
    @test SymbolicAnalysis.getproperty(info_feat1, :var) == :temperature
    @test SymbolicAnalysis.propertynames(info_feat1)     == (:id, :feat, :var, :nwin)
    @test SymbolicAnalysis.feature_id(info_feat1)        == 1
    @test SymbolicAnalysis.variable_name(info_feat1)     == :temperature
    @test SymbolicAnalysis.feature_type(info_feat1)      == :mean
    @test SymbolicAnalysis.window_number(info_feat1)     == 2
    
    # Test argument validation
    @test_throws ArgumentError SymbolicAnalysis.InfoFeat(5, :pressure, :std, 0)
    @test_throws ArgumentError SymbolicAnalysis.InfoFeat(6, :pressure, :std, -1)
end
