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

# ---------------------------------------------------------------------------- #
#                                 mas_caller                                   #
# ---------------------------------------------------------------------------- #
function mas_caller(ds::EitherDataSet, association::AbstractAssociationExtractor)
    X = if ds isa ModalDataSet
        get_logiset(ds)
    else
        _X = get_X(ds)
        _Xcol = ncol(_X)
        scalarlogiset(_X;
            relations=AbstractRelation[],
            conditions=Vector{ScalarMetaCondition}(
                collect(Iterators.flatten([
                    [ScalarMetaCondition(f, ≤) for f in VariableMin.(1:_Xcol)],
                    [ScalarMetaCondition(f, ≥) for f in VariableMin.(1:_Xcol)],
                    [ScalarMetaCondition(f, ≤) for f in VariableMax.(1:_Xcol)],
                    [ScalarMetaCondition(f, ≥) for f in VariableMax.(1:_Xcol)],
                ]))
            )
        )
    end

    algo      = get_method(association)
    masargs   = get_mas_args(association)
    maskwargs = get_mas_kwargs(association)

    miner = Miner(X, algo, masargs...; maskwargs...)
    mine!(miner)

    return arules(miner)
end
