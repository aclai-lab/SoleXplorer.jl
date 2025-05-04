using Test
using SoleXplorer

@testset "Base Interface Abstract Types" begin
    # Test all abstract types are defined and have the expected supertype
    @test SoleXplorer.AbstractDatasetSetup <: Any
    @test SoleXplorer.AbstractIndexCollection <: Any
    @test SoleXplorer.AbstractDataset <: Any
    @test SoleXplorer.AbstractModelType <: Any
    @test SoleXplorer.AbstractModelSetup{<:SoleXplorer.AbstractModelType} <: Any
    @test SoleXplorer.AbstractModelset{<:SoleXplorer.AbstractModelType} <: Any
    @test SoleXplorer.AbstractResults <: Any
    @test SoleXplorer.AbstractTypeParams <: Any
    
    # Define minimal concrete implementations for testing modeltype function
    struct TestModelType <: SoleXplorer.AbstractModelType end
    
    struct TestModelSetup <: SoleXplorer.AbstractModelSetup{TestModelType} end
    
    struct TestModelset <: SoleXplorer.AbstractModelset{TestModelType} end
    
    # Test modeltype function
    test_setup = TestModelSetup()
    test_modelset = TestModelset()
    
    @test SoleXplorer.modeltype(test_setup) == TestModelType
    @test SoleXplorer.modeltype(test_modelset) == TestModelType
    
    # Test parametric type constraints
    struct GenericModelType{T} <: SoleXplorer.AbstractModelType end
    struct GenericModelSetup{T} <: SoleXplorer.AbstractModelSetup{GenericModelType{T}} end
    struct GenericModelset{T} <: SoleXplorer.AbstractModelset{GenericModelType{T}} end
    
    generic_setup = GenericModelSetup{Float64}()
    generic_modelset = GenericModelset{Float64}()
    
    @test SoleXplorer.modeltype(generic_setup) == GenericModelType{Float64}
    @test SoleXplorer.modeltype(generic_modelset) == GenericModelType{Float64}
end