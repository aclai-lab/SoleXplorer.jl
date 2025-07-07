# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractSource <: MLJType end

# ---------------------------------------------------------------------------- #
#                                   source                                     #
# ---------------------------------------------------------------------------- #
# 'Source' wrappers for storing data as arguments.
# inspired by MLJ's `Source` interface, but simplified for Sole.
struct TableSource{T<:DataFrame} <: AbstractSource
    data :: T
end

struct VectorSource{S<:Label, T<:AbstractVector{S}} <: AbstractSource
    data :: T
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function source end

source(X::T) where {T<:DataFrame} = TableSource{T}(X)
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