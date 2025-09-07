# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const RangeSpec = Union{
    Tuple,
    Tuple{Vararg{Tuple}},
    Vector{<:MLJ.NumericRange},
    MLJBase.NominalRange
}

# ---------------------------------------------------------------------------- #
#                               Range adapter                                  #
# ---------------------------------------------------------------------------- #
make_mlj_ranges(range, model)
range = tuning.range isa Tuple{Vararg{Tuple}} ? tuning.range : (tuning.range,)
range = collect(MLJ.range(model, r[1]; r[2:end]...) for r in range)

# wrapper for MLJ.range
Base.range(field::Union{Symbol,Expr}; kwargs...) = field, kwargs...