# ---------------------------------------------------------------------------- #
#                         abstract types definition                            #
# ---------------------------------------------------------------------------- #
"""
Abstract types for dataset struct
"""
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
abstract type AbstractMeasures end

"""
Abstract type for type/params structs
"""
abstract type AbstractTypeParams end

# ---------------------------------------------------------------------------- #
#                               type aliases                                   #
# ---------------------------------------------------------------------------- #
const Optional{T}       = Union{T, Nothing}

const NamedTupleBool    = Union{NamedTuple, Bool}

const SymbolString      = Union{Symbol,AbstractString}
const Rule              = Union{DecisionList, DecisionEnsemble, DecisionSet}

const OptModel          = Optional{MLJ.Model}

const OptSymbol         = Optional{Symbol}
const OptTuple          = Optional{Tuple}
const OptVecTuple       = Optional{Vector{<:Tuple}}
const OptNamedTuple     = Optional{NamedTuple}
const OptCallable       = Optional{<:Base.Callable}
const OptVecCall        = Optional{Vector{<:Base.Callable}}
const OptDataType       = Optional{DataType}

const OptVector         = Optional{AbstractVector}
const OptStringVec      = Optional{Vector{<:SymbolString}}
const OptVecAbsModel    = Optional{Vector{<:AbstractModel}}
const OptVecMeas        = Optional{AbstractVector{<:MLJBase.StatisticalMeasuresBase.Wrapper}}
const OptAbsMeas        = Optional{AbstractMeasures}
const OptRules          = Optional{Rule}

# const Cat_Value = Union{AbstractString, Symbol, MLJ.CategoricalValue}
# const Reg_Value = Number
# const Y_Value   = Union{Cat_Value, Reg_Value}
