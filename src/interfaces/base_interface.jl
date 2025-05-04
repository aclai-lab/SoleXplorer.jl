# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
Abstract type for dataset struct
"""
abstract type AbstractDatasetSetup end
abstract type AbstractIndexCollection end
abstract type AbstractDataset end

"""
Abstract type for model type
"""
abstract type AbstractModelType end

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