# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractSource <: MLJType end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const NUMERIC_TYPE = Union{Number, Missing}

# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
is_numeric_dataframe(X::AbstractDataFrame) = all(T -> T <: NUMERIC_TYPE, eltype.(eachcol(X)))

# ---------------------------------------------------------------------------- #
#                                   source                                     #
# ---------------------------------------------------------------------------- #
# 'Source' wrappers for storing data as arguments.
# inspired by MLJ's `Source` interface, but simplified for Sole.
# and extended to support time series and vector data.
struct TableSource{T<:AbstractDataFrame} <: AbstractSource
    data :: T
end

struct VectorSource{S<:Label, T<:AbstractVector{S}} <: AbstractSource
    data :: T
end

struct TimeSeriesSource{T<:AbstractDataFrame} <: AbstractSource
    data      :: T
    params    :: NamedTuple

    function TimeSeriesSource(X::T, params::NamedTuple) where T<:AbstractDataFrame
        data = process_multidim_ds(X, params)
        new{T}(data, params)
    end
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function source end

function source(X::T, ts_params::NamedTuple) where {T<:AbstractDataFrame}
    is_numeric_dataframe(X) ?
        TableSource{T}(X) :
        TimeSeriesSource(X, ts_params)
end
source(X::T) where {S, T<:AbstractVector{S}} = VectorSource{S,T}(X)

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
nrows_at_source(X::TableSource)  = nrow(X.data)
ncols_at_source(X::TableSource)  = ncol(X.data)
nrows_at_source(X::VectorSource) = length(X.data)

# select rows in a TableSource
# examples: ts(rows=1:10), ts(rows=:) -> select all rows
function (X::TableSource)(; rows=:)
    rows == (:) && return X.data
    return @views X.data[rows, :]
end
# select elements in a VectorSource
# kepts 'rows' for consistency
function (X::VectorSource)(; rows=:)
    rows == (:) && return X.data
    return @views X.data[rows]
end

Base.isempty(X::AbstractSource)  = isempty(X.data)
color(::AbstractSource) = :yellow

function Base.show(io::IO, source::TableSource{T}) where T
    nrows = nrows_at_source(source)
    nclos = ncols_at_source(source)
    print(io, "TableSource{$T}($nrows x $nclos)")
end

function Base.show(io::IO, source::VectorSource{S,T}) where {S,T}
    nrows = nrows_at_source(source)
    print(io, "VectorSource{$S,$T}(length=$nrows)")
end