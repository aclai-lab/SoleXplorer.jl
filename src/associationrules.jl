# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractAssociationExtractor end

# ---------------------------------------------------------------------------- #
#                                  adapters                                    #
# ---------------------------------------------------------------------------- #
"""
Association rule mining algorithm extractors.

These types serve as adapters that wrap the ModalAssociationRules mining algorithms
(`apriori`, `fpgrowth`, `eclat`) to provide a unified interface within SoleXplorer.
"""
struct Apriori <: AbstractAssociationExtractor
    method :: Base.Callable
    args   :: Tuple
    kwargs :: Union{Nothing, NamedTuple}

    function Apriori(args...; kwargs...)
        isnothing(kwargs) ? 
            new(apriori, args, nothing) :
            new(apriori, args, NamedTuple(kwargs))
    end
end

struct FPGrowth <: AbstractAssociationExtractor
    method :: Base.Callable
    args   :: Tuple
    kwargs :: Union{Nothing, NamedTuple}

    function FPGrowth(args...; kwargs...)
        isnothing(kwargs) ? 
            new(fpgrowth, args, nothing) :
            new(fpgrowth, args, NamedTuple(kwargs))
    end
end

struct Eclat <: AbstractAssociationExtractor
    method :: Base.Callable
    args   :: Tuple
    kwargs :: Union{Nothing, NamedTuple}

    function Eclat(args...; kwargs...)
        isnothing(kwargs) ? 
            new(eclat, args, nothing) :
            new(eclat, args, NamedTuple(kwargs))
    end
end

get_method(a::AbstractAssociationExtractor)     = a.method
get_mas_args(a::AbstractAssociationExtractor)   = a.args
get_mas_kwargs(a::AbstractAssociationExtractor) = a.kwargs



# function associationrules(method::Base.Callable, args...; kwargs...)
#   ds = Miner(m, args...; kwargs...)
#   return ds
# end
