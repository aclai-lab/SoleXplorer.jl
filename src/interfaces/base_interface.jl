# ---------------------------------------------------------------------------- #
#                           const types definition                             #
# ---------------------------------------------------------------------------- #
const Optional{T}       = Union{T, Nothing}

const NamedTupleBool    = Union{NamedTuple, Bool}
const VecOrSubArray{T}  = Union{SubArray{T}, Vector{<:SubArray{T}}} where T
const VecOrMatrix       = Union{AbstractMatrix, Vector{<:AbstractMatrix}}

const SymbolString      = Union{Symbol,AbstractString}

const OptSymbol         = Optional{Symbol}
const OptTuple          = Optional{Tuple}
const OptNamedTuple     = Optional{NamedTuple}
const OptCallable       = Optional{<:Base.Callable}
const OptDataType       = Optional{DataType}

const OptStringVec      = Optional{Vector{<:AbstractString}}

const Cat_Value = Union{AbstractString, Symbol, MLJ.CategoricalValue}
const Reg_Value = Number
const Y_Value   = Union{Cat_Value, Reg_Value}

# ---------------------------------------------------------------------------- #
#                         abstract types definition                            #
# ---------------------------------------------------------------------------- #
"""
Abstract types for dataset struct
"""
abstract type AbstractDatasetSetup end
abstract type AbstractIndexCollection end
abstract type AbstractDataset end

"""
Abstract type for model type
"""
abstract type AbstractModelType end

"""
Abstract type for all classification models
"""
abstract type AbstractClassification <: AbstractModelType end

"""
Abstract type for all regression models.
"""
abstract type AbstractRegression     <: AbstractModelType end

"""
Abstract type for model configuration and parameters
"""
abstract type AbstractModelSetup{T<:AbstractModelType} end

modeltype(::AbstractModelSetup{T}) where {T} = T

"""
Abstract type for fitted model configurations
"""
abstract type AbstractModelset{T<:AbstractModelType} end

modeltype(::AbstractModelset{T}) where {T} = T

"""
Abstract type for results output
"""
abstract type AbstractResults end

"""
Abstract type for type/params structs
"""
abstract type AbstractTypeParams end