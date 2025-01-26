# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
check_dataframe_type(df::AbstractDataFrame) = all(col -> eltype(col) <: Union{Real, AbstractArray{<:Real}}, eachcol(df))
hasnans(X::AbstractDataFrame) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))

# ---------------------------------------------------------------------------- #
#                                 treatment                                    #
# ---------------------------------------------------------------------------- #
function _treatment(
    X::DataFrame,
    vnames::AbstractVector{String},
    treatment::Symbol,
    features::AbstractVector{<:Base.Callable},
    winparams::NamedTuple
)
    # check parameters
    haskey(winparams, :type) || throw(ArgumentError("winparams must contain a type, $(keys(WIN_PARAMS))"))
    haskey(WIN_PARAMS, winparams.type) || throw(ArgumentError("winparams.type must be one of: $(keys(WIN_PARAMS))"))

    max_interval = maximum(length.(eachrow(X)))
    _wparams = winparams |> x -> @delete x.type
    n_intervals = winparams.type(max_interval; _wparams...)
    
    if treatment == :aggregate        # propositional
        if n_intervals == 1
            valid_X = DataFrame([v => Float64[] 
                for v in [string(f, "(", v, ")") 
                for f in features for v in vnames]]
            )
        else
            valid_X = DataFrame([v => Float64[] 
                for v in [string(f, "(", v, ")w", i) 
                for f in features for v in vnames 
                for i in 1:length(n_intervals)]]
            )
        end

    elseif treatment == :reducesize   # modal
        valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])
    # else
    #     throw(ArgumentError("Treatments supported, :aggregate and :reducesize"))
    end

    for row in eachrow(X)
        row_intervals = winparams.type(maximum(length.(collect(row))); _wparams...)
        interval_diff = length(n_intervals) - length(row_intervals)

        if treatment == :aggregate
            push!(valid_X, vcat([
                    vcat([f(col[r]) for r in row_intervals], 
                    fill(NaN, interval_diff)) for col in row, f in features
                ]...)
        )
        elseif treatment == :reducesize
            f = haskey(_wparams, :reducefunc) ? _wparams.reducefunc : mean
            push!(valid_X, [
                    vcat([f(col[r]) for r in row_intervals], 
                    fill(NaN, interval_diff)) for col in row
                ]
            )
        end
    end

    return valid_X
end

# ---------------------------------------------------------------------------- #
#                                 partitioning                                 #
# ---------------------------------------------------------------------------- #
function _partition(
    y::Union{CategoricalArray, Vector{T}},
    train_ratio::Float64,
    shuffle::Bool,
    stratified_sampling::Bool,
    nfolds::Int,
    rng::AbstractRNG
) where {T<:Union{AbstractString, Number}}
    if stratified_sampling
        stratified_cv = MLJ.StratifiedCV(; nfolds, shuffle, rng)
        tt = MLJ.MLJBase.train_test_pairs(stratified_cv, 1:length(y), y)
        return [TT_indexes(train, test) for (train, test) in tt]
    else
        return TT_indexes(MLJ.partition(eachindex(y), train_ratio; shuffle, rng)...)
    end
end

# ---------------------------------------------------------------------------- #
#                               prepare dataset                                #
# ---------------------------------------------------------------------------- #
# TODO automatizzare classification e regression in base alla presenza di y? verificare i dataset
function prepare_dataset(
    X::AbstractDataFrame,
    y::AbstractVector;
    # model.config
    algo::Symbol=:classification,
    treatment::Symbol=:aggregate,
    features::AbstractVector{<:Base.Callable}=DEFAULT_FEATS,
    # model.preprocess
    train_ratio::Float64=0.8,
    shuffle::Bool=true,
    stratified_sampling::Bool=false,
    nfolds::Int=6,
    rng::AbstractRNG=Random.TaskLocalRNG(),
    # model.winparams
    winparams::Union{NamedTuple, Nothing}=nothing,
    vnames::Union{AbstractVector{<:Union{AbstractString, Symbol}}, Nothing}=nothing,
)
    # check parameters
    check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))
    treatment in AVAIL_TREATMENTS || throw(ArgumentError("Treatment must be one of: $AVAIL_TREATMENTS"))

    if algo == :regression
        y isa AbstractVector{<:Number} || throw(ArgumentError("Regression requires a numeric target variable"))
        y isa AbstractFloat || (y = Float64.(y))
    elseif algo == :classification
        y isa CategoricalArray || (y = CategoricalArray(y))
    else
        throw(ArgumentError("Algorithms supported, :regression and :classification"))
    end

    if isnothing(vnames)
        vnames = names(X)
    else
        size(X, 2) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
        vnames = eltype(vnames) <: Symbol ? string.(vnames) : vnames
    end

    hasnans(X) && @warn "DataFrame contains NaN values"

    column_eltypes = eltype.(eachcol(X))

    # case 1: dataframe with numeric columns
    if all(t -> t <: Number, column_eltypes)
        # dataframe with numeric columns
        return SoleXplorer.Dataset(
            DataFrame(vnames .=> eachcol(X)), y,
            _partition(y, train_ratio, shuffle, stratified_sampling, nfolds, rng)
        )
    # case 2: dataframe with vector-valued columns
    elseif all(t -> t <: AbstractVector{<:Number}, column_eltypes)
        # dataframe with vector-valued columns
        return SoleXplorer.Dataset(
            # if winparams is nothing, then leave the dataframe as it is
            isnothing(winparams) ? DataFrame(vnames .=> eachcol(X)) : _treatment(X, vnames, treatment, features, winparams), y,
            _partition(y, train_ratio, shuffle, stratified_sampling, nfolds, rng)
        )
    else
        throw(ArgumentError("Column type not yet supported"))
    end
end

function prepare_dataset(
    X::AbstractDataFrame, 
    y::AbstractVector, 
    model::AbstractModelSet
)
    prepare_dataset(
        X, y; 
        algo                = model.config.algo,
        treatment           = model.config.treatment,
        features            = model.features,
        # model.preprocess
        train_ratio         = model.preprocess.train_ratio,
        shuffle             = model.preprocess.shuffle,
        stratified_sampling = model.preprocess.stratified_sampling,
        nfolds              = model.preprocess.nfolds,
        rng                 = model.preprocess.rng,
        winparams           = model.winparams,
    )
end

# y is not a vector, but a symbol or a string that identifies the column in X
function prepare_dataset(
    X::AbstractDataFrame, 
    y::Union{Symbol, AbstractString}, 
    args...; kwargs...
)
    prepare_dataset(X[!, Not(y)], X[!, y], args...; kwargs...)
end
