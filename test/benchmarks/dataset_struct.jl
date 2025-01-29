using BenchmarkTools
using Sole, MLJ
using DataFrames

X, y = Sole.load_arff_dataset("NATOPS")

struct TT_indexes
    # train       :: AbstractVector{<:Int}
    # test        :: AbstractVector{<:Int}
    train       :: Vector{Int}
    test        :: Vector{Int}
end

Base.show(io::IO, t::TT_indexes) = print(io, "TT_indexes(train=", t.train, ", test=", t.test, ")")

struct Dataset_v1{T<:AbstractDataFrame,S}
    X           :: T
    y           :: S
    tt          :: Union{TT_indexes, AbstractVector{<:TT_indexes}}
end

struct Dataset_v2{T<:AbstractDataFrame,S}
    X           :: T
    y           :: S
    tt          :: Union{TT_indexes, AbstractVector{<:TT_indexes}}
    Xtrain      :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    Xtest       :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}

    function Dataset_v2(X::T, y::S, tt) where {T<:AbstractDataFrame,S}
        Xtrain, Xtest, ytrain, ytest = 
            (
            view(X, tt.train, :),
            view(X, tt.test, :),
            view(y, tt.train),
            view(y, tt.test)
            )

        new{T,S}(X, y, tt, Xtrain, Xtest, ytrain, ytest)
    end
end

struct Dataset_v3{T<:AbstractDataFrame,S}
    X           :: T
    y           :: S
    tt          :: Union{TT_indexes, AbstractVector{<:TT_indexes}}
    Xtrain      :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    Xtest       :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}

    function Dataset_v3(X::T, y::S, tt) where {T<:AbstractDataFrame,S}
        Xtrain = @views X[tt.train, :]
        Xtest =  @views X[tt.test, :]
        ytrain = @views y[tt.train]
        ytest = @views y[tt.test]

        new{T,S}(X, y, tt, Xtrain, Xtest, ytrain, ytest)
    end
end

using DataFrames

# Overload show so we can inspect TT_indexes easily
Base.show(io::IO, t::TT_indexes) = print(io, "TT_indexes(train=", t.train, ", test=", t.test, ")")

"""
Dataset_v4 can store DataFrames and corresponding label vectors, along with
train/test indexes. It supports single or multiple TT_indexes objects.

Fields:
- X, y: The original DataFrame and label vector.
- tt: A single TT_indexes or a collection of TT_indexes.
- Xtrain, Xtest, ytrain, ytest: Subsetted training/testing dataframes and labels, 
  either single or vectors of the same length as tt.
"""
struct Dataset_v4{T<:AbstractDataFrame,S}
    X           :: T
    y           :: S
    tt          :: Union{TT_indexes, AbstractVector{<:TT_indexes}}
    Xtrain      :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    Xtest       :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
end

"""
Main constructor for Dataset_v4. Dispatches on whether `tt` is a single TT_indexes
or a collection of TT_indexes.
"""
function Dataset_v4(X::T, y::S, tt::TT_indexes) where {T<:AbstractDataFrame, S}
    @assert size(X, 1) == length(y) "Mismatch between row count of X and length of y."
    Xtrain = @views X[tt.train, :]
    Xtest  = @views X[tt.test, :]
    ytrain = @views y[tt.train]
    ytest  = @views y[tt.test]

    Dataset_v4{T,S}(X, y, tt, Xtrain, Xtest, ytrain, ytest)
end

function Dataset_v4(X::T, y::S, tt::AbstractVector{<:TT_indexes}) where {T<:AbstractDataFrame, S}
    @assert size(X, 1) == length(y) "Mismatch between row count of X and length of y."
    # For each TT_indexes in the vector, produce a sub-dataframe and sub-label vector
    Xtrain_vec = Vector{SubDataFrame{T}}()
    Xtest_vec  = Vector{SubDataFrame{T}}()
    ytrain_vec = Vector{SubArray{<:eltype(S)}}()
    ytest_vec  = Vector{SubArray{<:eltype(S)}}()

    for fold in tt
        push!(Xtrain_vec, @views X[fold.train, :])
        push!(Xtest_vec,  @views X[fold.test,  :])
        push!(ytrain_vec, @views y[fold.train])
        push!(ytest_vec,  @views y[fold.test])
    end

    Dataset_v4{T,S}(X, y, tt, Xtrain_vec, Xtest_vec, ytrain_vec, ytest_vec)
end

train_ratio = 0.8
tt = TT_indexes(MLJ.partition(eachindex(y), train_ratio)...)

# ---------------------------------------------------------------------------- #
#                                   version 1                                  #
# ---------------------------------------------------------------------------- #
# @btime ds_v1 = Dataset_v1(X, y, tt);

# ds_v1 = Dataset_v1(X, y, tt);
# @btime ds_v1.X[ds_v1.tt.train, :]

println("Version 1:")
@btime begin 
    ds_v1 = Dataset_v1(X, y, tt);
    @views ds_v1.X[ds_v1.tt.train, :]
    @views ds_v1.X[ds_v1.tt.test, :]
    @views ds_v1.y[ds_v1.tt.train]
    @views ds_v1.y[ds_v1.tt.test]
end

# ---------------------------------------------------------------------------- #
#                                   version 2                                  #
# ---------------------------------------------------------------------------- #
# @btime ds_v2 = Dataset_v2(X, y, tt);

# ds_v2 = Dataset_v2(X, y, tt);
# @btime ds_v2.Xtrain

println("Version 2:")
@btime begin 
    ds_v2 = Dataset_v2(X, y, tt);
    ds_v2.Xtrain
    ds_v2.Xtest
    ds_v2.ytrain
    ds_v2.ytest
end

# ---------------------------------------------------------------------------- #
#                                   version 3                                  #
# ---------------------------------------------------------------------------- #
# @btime ds_v3 = Dataset_v3(X, y, tt);

# ds_v3 = Dataset_v3(X, y, tt);
# @btime ds_v3.Xtrain

println("Version 3:")
@btime begin 
    ds_v3 = Dataset_v3(X, y, tt);
    ds_v3.Xtrain
    ds_v3.Xtest
    ds_v3.ytrain
    ds_v3.ytest
end

# ---------------------------------------------------------------------------- #
#                                   version 4                                  #
# ---------------------------------------------------------------------------- #
# @btime ds_v4 = Dataset_v4(X, y, tt);

# ds_v4 = Dataset_v4(X, y, tt);
# @btime ds_v4.Xtrain

println("Version 4:")
@btime begin 
    ds_v4 = Dataset_v4(X, y, tt);
    ds_v4.Xtrain
    ds_v4.Xtest
    ds_v4.ytrain
    ds_v4.ytest
end