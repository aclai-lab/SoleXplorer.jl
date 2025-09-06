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
#                             Range normalization                              #
# ---------------------------------------------------------------------------- #
normalize_range(range::Union{Vector{<:MLJ.NumericRange}, MLJBase.NominalRange}) = range
normalize_range(range::Tuple{Vararg{Tuple}}) = range
normalize_range(range::Tuple) = (range,)