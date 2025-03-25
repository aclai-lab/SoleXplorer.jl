# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
check_dataset_type(df::AbstractDataFrame) = all(col -> eltype(col) <: Union{Real,AbstractArray{<:Real}}, eachcol(df))
check_dataset_type(X::AbstractMatrix) = eltype(X) <: Union{Real,AbstractArray{<:Real}}
hasnans(df::AbstractDataFrame) = any(x -> x == 1, SoleData.hasnans.(eachcol(df)))
hasnans(X::AbstractMatrix) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))

# ---------------------------------------------------------------------------- #
#                                 partitioning                                 #
# ---------------------------------------------------------------------------- #
"""
    _partition(y::Union{CategoricalArray, Vector{T}}, train_ratio::Float64, 
               shuffle::Bool, stratified::Bool, nfolds::Int, rng::AbstractRNG) 
               where {T<:Union{AbstractString, Number}}

Partitions the input vector `y` into training and testing indices based on 
the specified parameters. Supports both stratified and non-stratified 
partitioning.

# Arguments
- `y::Union{CategoricalArray, Vector{T}}`: The target variable to partition.
- `train_ratio::Float64`: The ratio of data to be used for training in 
  non-stratified partitioning.
- `shuffle::Bool`: Whether to shuffle the data before partitioning.
- `stratified::Bool`: Whether to perform stratified partitioning.
- `nfolds::Int`: Number of folds for cross-validation in stratified 
  partitioning.
- `rng::AbstractRNG`: Random number generator for reproducibility.

# Returns
- `Vector{Tuple{Vector{Int}, Vector{Int}}}`: A vector of tuples containing 
  training and testing indices.

# Throws
- `ArgumentError`: If `nfolds` is less than 2 when `stratified` is true.
"""

function _partition(
    y::Union{CategoricalArray,Vector{T}},
    # validation::Bool,
    train_ratio::Float64,
    valid_ratio::Float64,
    shuffle::Bool,
    stratified::Bool,
    nfolds::Int,
    rng::AbstractRNG
) where {T<:Union{AbstractString,Number}}
    if stratified
        stratified_cv = MLJ.StratifiedCV(; nfolds, shuffle, rng)
        tt = MLJ.MLJBase.train_test_pairs(stratified_cv, 1:length(y), y)
        if valid_ratio == 1.0
            return [TT_indexes(train, eltype(train)[], test) for (train, test) in tt]
        else
            tv = collect((MLJ.partition(t[1], train_ratio)..., t[2]) for t in tt)
            return [TT_indexes(train, valid, test) for (train, valid, test) in tv]
        end
    else
        tt = MLJ.partition(eachindex(y), train_ratio; shuffle, rng)
        if valid_ratio == 1.0
            return TT_indexes(tt[1], eltype(tt[1])[], tt[2])
        else
            tv = MLJ.partition(tt[1], valid_ratio; shuffle, rng)
            return TT_indexes(tv[1], tv[2], tt[2])
        end
    end
end

# ---------------------------------------------------------------------------- #
#                               prepare dataset                                #
# ---------------------------------------------------------------------------- #
"""
    prepare_dataset(X::AbstractDataFrame, y::AbstractVector; algo::Symbol=:classification, 
                    treatment::Symbol=:aggregate, features::AbstractVector{<:Base.Callable}=DEFAULT_FEATS, 
                    train_ratio::Float64=0.8, shuffle::Bool=true, stratified::Bool=false, 
                    nfolds::Int=6, rng::AbstractRNG=Random.TaskLocalRNG(), 
                    winparams::Union{NamedTuple,Nothing}=nothing, 
                    vnames::Union{AbstractVector{<:Union{AbstractString,Symbol}},Nothing}=nothing)

Prepares a dataset for machine learning by processing the input DataFrame `X` and target vector `y`. 
Supports both classification and regression tasks, with options for data treatment and partitioning.

# Arguments
- `X::AbstractDataFrame`: The input data containing features.
- `y::AbstractVector`: The target variable corresponding to the rows in `X`.
- `algo::Symbol`: The type of algorithm, either `:classification` or `:regression`.
- `treatment::Symbol`: The data treatment method, default is `:aggregate`.
- `features::AbstractVector{<:Base.Callable}`: Functions to apply to the data columns.
- `train_ratio::Float64`: Ratio of data to be used for training.
- `shuffle::Bool`: Whether to shuffle data before partitioning.
- `stratified::Bool`: Whether to use stratified partitioning.
- `nfolds::Int`: Number of folds for cross-validation.
- `rng::AbstractRNG`: Random number generator for reproducibility.
- `winparams::Union{NamedTuple,Nothing}`: Parameters for windowing strategy.
- `vnames::Union{AbstractVector{<:Union{AbstractString,Symbol}},Nothing}`: Names of the columns in `X`.

# Returns
- `SoleXplorer.Dataset`: A dataset object containing processed data and partitioning information.

# Throws
- `ArgumentError`: If input parameters are invalid or unsupported column types are encountered.
"""

function prepare_dataset(
    X::AbstractDataFrame,
    y::AbstractVector;
    # model.config
    algo::Symbol=:classification,
    treatment::Symbol=:aggregate,
    features::AbstractVector{<:Base.Callable}=DEFAULT_FEATS,
    # validation::Bool=false,
    # model.preprocess
    train_ratio::Float64=0.8,
    valid_ratio::Float64=1.0,
    shuffle::Bool=true,
    stratified::Bool=false,
    nfolds::Int=6,
    rng::AbstractRNG=Random.TaskLocalRNG(),
    # model.winparams
    winparams::SoleFeatures.WinParams,
    vnames::Union{AbstractVector{<:Union{AbstractString,Symbol}},Nothing}=nothing,
)
    # check parameters
    check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))
    treatment in AVAIL_TREATMENTS || throw(ArgumentError("Treatment must be one of: $AVAIL_TREATMENTS"))

    if algo == :regression
        y isa AbstractVector{<:Number} || throw(ArgumentError("Regression requires a numeric target variable"))
        y isa AbstractFloat || (y = Float64.(y))
    elseif algo == :classification
        y isa AbstractVector{<:AbstractFloat} && throw(ArgumentError("Classification requires a categorical target variable"))
        y isa CategoricalArray || (y = coerce(y, MLJ.Multiclass))
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

    ds_info = DatasetInfo(
        algo,
        treatment,
        features,
        train_ratio,
        valid_ratio,
        shuffle,
        stratified,
        nfolds,
        rng,
        winparams,
        vnames,
        # validation
    )

    # case 1: dataframe with numeric columns
    if all(t -> t <: Number, column_eltypes)
        return SoleXplorer.Dataset(
            DataFrame(vnames .=> eachcol(X)), y,
            _partition(y, train_ratio, valid_ratio, shuffle, stratified, nfolds, rng),
            ds_info
        )
    # case 2: dataframe with vector-valued columns
    elseif all(t -> t <: AbstractVector{<:Number}, column_eltypes)
        return SoleXplorer.Dataset(
            # if winparams is nothing, then leave the dataframe as it is
            isnothing(winparams) ? DataFrame(vnames .=> eachcol(X)) : 
                SoleFeatures._treatment(X, vnames, treatment, features, winparams), y,
            _partition(y, train_ratio, valid_ratio, shuffle, stratified, nfolds, rng),
            ds_info
        )
    else
        throw(ArgumentError("Column type not yet supported"))
    end
end

function prepare_dataset(
    X::AbstractDataFrame,
    y::AbstractVector,
    model::AbstractModelSetup
)
    # check if it's needed also validation set
    # validation = haskey(VALIDATION, model.type) && getproperty(model.params, VALIDATION[model.type][1]) != VALIDATION[model.type][2]
    # valid_ratio = (validation && model.preprocess.valid_ratio == 1) ? 0.8 : model.preprocess.valid_ratio

    prepare_dataset(
        X, y;
        algo=model.config.algo,
        treatment=model.config.treatment,
        features=model.features,
        # validation,
        # model.preprocess
        train_ratio=model.preprocess.train_ratio,
        valid_ratio=model.preprocess.valid_ratio,
        shuffle=model.preprocess.shuffle,
        stratified=model.preprocess.stratified,
        nfolds=model.preprocess.nfolds,
        rng=model.preprocess.rng,
        winparams=model.winparams,
    )
end

# y is not a vector, but a symbol or a string that identifies the column in X
function prepare_dataset(
    X::AbstractDataFrame,
    y::Union{Symbol,AbstractString},
    args...; kwargs...
)
    prepare_dataset(X[!, Not(y)], X[!, y], args...; kwargs...)
end
