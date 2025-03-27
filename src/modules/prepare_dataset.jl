# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
check_dataset_type(df::AbstractDataFrame) = all(col -> eltype(col) <: Union{Real,AbstractArray{<:Real}}, eachcol(df))
check_dataset_type(X::AbstractMatrix) = eltype(X) <: Union{Real,AbstractArray{<:Real}}
hasnans(df::AbstractDataFrame) = any(x -> x == 1, SoleData.hasnans.(eachcol(df)))
hasnans(X::AbstractMatrix) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))

"""
    check_row_consistency(X::AbstractMatrix) -> Bool

Check that all elements within each row of the matrix have consistent dimensions.
This is important for time series or array data where operations expect 
columns within a row to have matching dimensions.
"""
function check_row_consistency(X::AbstractMatrix) 
    for row in eachrow(X)
        # skip rows with only scalar values
        any(el -> el isa AbstractArray, row) || continue
        
        # find first array element to use as reference
        ref_idx = findfirst(el -> el isa AbstractArray, row)
        isnothing(ref_idx) && continue
        
        ref_size = size(row[ref_idx])
        
        # check if any array element has different size (short-circuit)
        if any(row) do el
                el isa AbstractArray && size(el) != ref_size
            end
            return false
        end
    end
    return true
end
# ---------------------------------------------------------------------------- #
#                                 partitioning                                 #
# ---------------------------------------------------------------------------- #
"""
    _partition(y::AbstractVector{<:Y_Value}, train_ratio::Float64, valid_ratio::Float64, 
               shuffle::Bool, stratified::Bool, nfolds::Int, rng::AbstractRNG)
               -> Union{TT_indexes, Vector{TT_indexes}}

Partitions the input vector `y` into training, validation, and testing indices based on 
the specified parameters. Supports both stratified and non-stratified partitioning.

# Arguments
- `y::AbstractVector{<:Y_Value}`: The target variable to partition.
- `train_ratio::Float64`: The ratio of data to be used for training (from the non-test portion when valid_ratio < 1.0).
- `valid_ratio::Float64`: Controls validation set creation:
  - When 1.0: No validation set is created (empty array)
  - When < 1.0: Creates validation set as a portion of the training data
- `shuffle::Bool`: Whether to shuffle the data before partitioning.
- `stratified::Bool`: Whether to use stratified partitioning:
  - When true: Returns a vector of TT_indexes for cross-validation
  - When false: Returns a single TT_indexes instance
- `nfolds::Int`: Number of folds for cross-validation in stratified partitioning.
- `rng::AbstractRNG`: Random number generator for reproducibility.

# Returns
- `Union{TT_indexes, Vector{TT_indexes}}`: Either:
  - A single `TT_indexes` object containing train/valid/test indices (when stratified=false)
  - A vector of `TT_indexes` objects for cross-validation folds (when stratified=true)
"""

function _partition(
    y::AbstractVector{<:Y_Value},
    train_ratio::Float64,
    valid_ratio::Float64,
    resample::Union{Resample, Nothing},
    rng::AbstractRNG
)::Union{TT_indexes{Int}, Vector{TT_indexes{Int}}}
    if isnothing(resample)
        tt = MLJ.partition(eachindex(y), train_ratio; shuffle=true, rng)
        if valid_ratio == 1.0
            return TT_indexes(tt[1], eltype(tt[1])[], tt[2])
        else
            tv = MLJ.partition(tt[1], valid_ratio; shuffle, rng)
            return TT_indexes(tv[1], tv[2], tt[2])
        end
    else
        resample_cv = resample.type(; resample.params...)
        tt = MLJ.MLJBase.train_test_pairs(resample_cv, 1:length(y), y)
        if valid_ratio == 1.0
            return [TT_indexes(train, eltype(train)[], test) for (train, test) in tt]
        else
            tv = collect((MLJ.partition(t[1], train_ratio)..., t[2]) for t in tt)
            return [TT_indexes(train, valid, test) for (train, valid, test) in tv]
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

function _prepare_dataset(
    df::AbstractDataFrame,
    y::AbstractVector;
    algo::Symbol,
    treatment::Symbol,
    reducefunc::Union{Base.Callable, Nothing},
    features::AbstractVector{<:Base.Callable},
    train_ratio::Float64,
    valid_ratio::Float64,
    rng::AbstractRNG,
    resample::Union{Resample, Nothing},
    winparams::SoleFeatures.WinParams,
    vnames::Union{SoleFeatures.VarNames,Nothing}=nothing,
)::Dataset
    X = Matrix(df)
    # check parameters
    check_dataset_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))
    check_row_consistency(X) || throw(ArgumentError("Elements within each row must have consistent dimensions"))
    treatment in AVAIL_TREATMENTS || throw(ArgumentError("Treatment must be one of: $AVAIL_TREATMENTS"))

    if algo == :regression
        y isa AbstractVector{<:Reg_Value} || throw(ArgumentError("Regression requires a numeric target variable"))
        y isa AbstractFloat || (y = Float64.(y))
    elseif algo == :classification
        y isa AbstractVector{<:Cat_Value} || throw(ArgumentError("Classification requires a categorical target variable"))
        y isa CategoricalArray || (y = coerce(y, MLJ.Multiclass))
    else
        throw(ArgumentError("Algorithms supported, :regression and :classification"))
    end

    if isnothing(vnames)
        vnames = names(df)
    else
        size(X, 2) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
        vnames isa AbstractVector{<:AbstractString} || (vnames = string.(vnames))
    end

    hasnans(X) && @warn "DataFrame contains NaN values"

    column_eltypes = eltype.(eachcol(X))

    isnothing(reducefunc) && (reducefunc = mean)

    if all(t -> t <: AbstractVector{<:Number}, column_eltypes) && !isnothing(winparams)
        X, vnames = SoleFeatures._treatment(X, vnames, treatment, features, winparams; reducefunc)
    end

    ds_info = DatasetInfo(
        algo,
        treatment,
        treatment == :reducesize ? reducefunc : nothing,
        train_ratio,
        valid_ratio,
        rng,
        isnothing(resample) ? false : true,
        vnames
    )

    return Dataset(
        X, y,
        # _partition(y, train_ratio, valid_ratio, shuffle, stratified, nfolds, rng),
        _partition(y, train_ratio, valid_ratio, resample, rng),
        ds_info
    )
end

function _prepare_dataset(
    X::AbstractDataFrame,
    y::AbstractVector,
    model::AbstractModelSetup
)::Dataset
    # modal reduce function, optional for propositional
    reducefunc = haskey(model.config, :reducefunc) ? model.config.reducefunc : nothing

    _prepare_dataset(
        X, y;
        algo=model.config.algo,
        treatment=model.config.treatment,
        reducefunc,
        features=model.features,
        train_ratio=model.preprocess.train_ratio,
        valid_ratio=model.preprocess.valid_ratio,
        rng=model.preprocess.rng,
        resample=model.resample,
        winparams=model.winparams,
    )
end

function prepare_dataset(
    X::AbstractDataFrame,
    y::AbstractVector;
    model::Union{NamedTuple, Nothing}=nothing,
    resample::Union{NamedTuple, Nothing}=nothing,
    win::Union{NamedTuple, Nothing}=nothing,
    features::Union{Tuple, Nothing}=nothing,
    tuning::Union{NamedTuple, Bool, Nothing}=nothing,
    rules::Union{NamedTuple, Nothing}=nothing,
    preprocess::Union{NamedTuple, Nothing}=nothing
)::Modelset
    # if model is unspecified, use default model setup
    isnothing(model) && (model = DEFAULT_MODEL_SETUP)
    modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, rules, preprocess)
    Modelset(modelset, _prepare_dataset(X, y, modelset))
end

# y is not a vector, but a symbol or a string that identifies the column in X
function prepare_dataset(
    X::AbstractDataFrame,
    y::Union{Symbol,AbstractString};
    kwargs...
)::Modelset
    prepare_dataset(X[!, Not(y)], X[!, y]; kwargs...)
end
