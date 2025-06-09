# ---------------------------------------------------------------------------- #
#                           const types definition                             #
# ---------------------------------------------------------------------------- #
const Optional{T}       = Union{T, Nothing}

const NamedTupleBool    = Union{NamedTuple, Bool}

const OptSymbol         = Optional{Symbol}
const OptTuple          = Optional{Tuple}
const OptNamedTuple     = Optional{NamedTuple}
# const OptNamedTupleBool = Optional{NamedTupleBool}
const OptCallable       = Optional{<:Base.Callable}
const OptDataType       = Optional{DataType}

# ---------------------------------------------------------------------------- #
#                         abstract types definition                            #
# ---------------------------------------------------------------------------- #
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
abstract type AbstractRegression <: AbstractModelType end

"""
Abstract type for dataset struct
"""
abstract type AbstractDatasetSetup{T<:AbstractModelType} end

modeltype(::AbstractDatasetSetup{T}) where {T} = T

abstract type AbstractIndexCollection end
abstract type AbstractDataset end

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