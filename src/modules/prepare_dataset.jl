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

"""
    code_dataset(X::AbstractDataFrame) -> AbstractDataFrame
    code_dataset(y::AbstractVector) -> AbstractVector
    code_dataset(X::AbstractDataFrame, y::AbstractVector) -> Tuple{AbstractDataFrame, AbstractVector}

Convert categorical/non-numeric data to numeric form by replacing categories with their level codes.

# Arguments
- `X::AbstractDataFrame`: DataFrame containing columns to be converted
- `y::AbstractVector`: Vector of target values to be converted

# Returns
- The input data with non-numeric values converted to their corresponding level codes

# Details
This function converts categorical or other non-numeric data to a numeric representation:
- For DataFrames: Each non-numeric column is replaced with integer level codes
- For vectors: Non-numeric elements are replaced with integer level codes
- When both X and y are provided, both are converted and returned as a tuple

Level codes maintain the original categorical information while allowing algorithms
that require numeric inputs to process the data.
"""
function code_dataset(X::AbstractDataFrame)
    for (name, col) in pairs(eachcol(X))
        if !(eltype(col) <: Number)
            X_coded = CategoricalArrays.levelcode.(categorical(col)) 
            X[!, name] = X_coded
        end
    end
    
    return X
end

function code_dataset(y::AbstractVector)
    if !(eltype(y) <: Number)
        eltype(y) <: Symbol && (y = string.(y))
        y = CategoricalArrays.levelcode.(categorical(y)) 
    end
    
    return y
end

code_dataset(X::AbstractDataFrame, y::AbstractVector) = code_dataset(X), code_dataset(y)

# ---------------------------------------------------------------------------- #
#                                 partitioning                                 #
# ---------------------------------------------------------------------------- #
"""
    _partition(y::AbstractVector{<:Y_Value}, train_ratio::Float64, valid_ratio::Float64, 
               resample::Union{Resample, Nothing}, rng::AbstractRNG)
               -> Union{TT_indexes{Int}, Vector{TT_indexes{Int}}}

Partitions the input vector `y` into training, validation, and testing indices based on 
the specified parameters. Supports both simple partitioning and cross-validation.

# Arguments
- `y::AbstractVector{<:Y_Value}`: The target variable to partition
- `train_ratio::Float64`: The ratio of data to be used for training (from the non-test portion)
- `valid_ratio::Float64`: Controls validation set creation:
  - When 1.0: No validation set is created (empty array)
  - When < 1.0: Creates a validation set as fraction of the training data
- `resample::Union{Resample, Nothing}`: Resampling strategy:
  - When `nothing`: Performs a single train/valid/test split
  - When a `Resample` object: Performs cross-validation according to the resampling strategy
- `rng::AbstractRNG`: Random number generator for reproducibility

# Returns
- `Union{TT_indexes{Int}, Vector{TT_indexes{Int}}}`: Either:
  - A single `TT_indexes{Int}` object with train/valid/test indices (when resample is `nothing`)
  - A vector of `TT_indexes{Int}` objects for cross-validation folds (when using resampling)

# Details
## Simple Partitioning (resample = nothing)
When `resample` is `nothing`, a single train/test split is created using `train_ratio`,
followed by a train/validation split of the training data using `valid_ratio`.

## Cross-Validation (resample != nothing)
When a resampling strategy is provided, the function:
1. Creates multiple train/test splits according to the strategy (e.g. k-fold CV)
2. For each fold, optionally splits the training data to create validation sets
3. Returns a vector of `TT_indexes` objects, one for each fold
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
            tv = MLJ.partition(tt[1], valid_ratio; shuffle=true, rng)
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
    prepare_dataset(X::AbstractDataFrame, y::AbstractVector; kwargs...)::Modelset
    prepare_dataset(X::AbstractDataFrame, y::Union{Symbol,AbstractString}; kwargs...)::Modelset

Prepares a dataset for machine learning by processing the input data and configuring a model setup.
Supports both classification and regression tasks, with extensive customization options.

# Arguments
- `X::AbstractDataFrame`: The input data containing features
- `y::AbstractVector` or `y::Union{Symbol,AbstractString}`: The target variable, either as a vector or 
  as a column name/symbol from `X`

# Optional Keyword Arguments
- `model::Union{NamedTuple, Nothing}=nothing`: Model configuration with fields:
  - `type`: Model type (e.g., `:xgboost`, `:randomforest`)
  - `params`: Model-specific parameters
- `resample::Union{NamedTuple, Nothing}=nothing`: Resampling strategy
  - `type`: Resampling method (e.g., `:cv`, `:stratifiedcv`)
  - `params`: Resampling parameters like `nfolds`
- `win::Union{NamedTuple, Nothing}=nothing`: Windowing parameters for time series data
  - `type`: Window function (e.g., `adaptivewindow`, `wholewindow`)
  - `params`: Window parameters like `nwindows`
- `features::Union{Tuple, Nothing}=nothing`: Statistical functions to extract from time series
  (e.g., `(mean, std, maximum)`)
- `tuning::Union{NamedTuple, Bool, Nothing}=nothing`: Hyperparameter tuning configuration
- `rules::Union{NamedTuple, Nothing}=nothing`: Rules for post-hoc explanation
- `preprocess::Union{NamedTuple, Nothing}=nothing`: Data preprocessing parameters:
  - `train_ratio`: Ratio of data for training vs testing
  - `valid_ratio`: Ratio of training data for validation
  - `rng`: Random number generator
- `reducefunc::Union{Base.Callable, Nothing}=nothing`: Function for reducing time series data
  in `:reducesize` treatment mode (default: `mean`)

# Returns
- `Modelset`: A complete modelset containing both the configured model setup and prepared dataset

# Notes
- For non-numeric data in `X`, use `code_dataset(X)` before calling this function
- Time series data should be stored as vectors within DataFrame cells
- If `y` is provided as a column name, it will be extracted from `X`
- When `model=nothing`, a default model setup is used
"""
function _prepare_dataset(
    df::AbstractDataFrame,
    y::AbstractVector;
    algo::Symbol,
    treatment::Symbol,
    features::AbstractVector{<:Base.Callable},
    train_ratio::Float64,
    valid_ratio::Float64,
    rng::AbstractRNG,
    resample::Union{Resample, Nothing},
    winparams::SoleFeatures.WinParams,
    vnames::Union{SoleFeatures.VarNames,Nothing}=nothing,
    reducefunc::Union{Base.Callable, Nothing}=nothing
)::Dataset
    X = Matrix(df)
    # check parameters
    check_dataset_type(X) || throw(ArgumentError("DataFrame must contain only numeric values, use SoleXplorer.code_dataset() to convert non-numeric data"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))
    check_row_consistency(X) || throw(ArgumentError("Elements within each row must have consistent dimensions"))
    # treatment in AVAIL_TREATMENTS || throw(ArgumentError("Treatment must be one of: $AVAIL_TREATMENTS"))

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

    if all(t -> t <: AbstractVector{<:Number}, column_eltypes) && !isnothing(winparams)
        X, vnames = SoleFeatures._treatment(X, vnames, treatment, features, winparams; reducefunc)
    end

    ds_info = DatasetInfo(
        algo,
        treatment,
        reducefunc,
        train_ratio,
        valid_ratio,
        rng,
        isnothing(resample) ? false : true,
        vnames
    )

    return Dataset(
        X, y,
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
    preprocess::Union{NamedTuple, Nothing}=nothing,
    reducefunc::Union{Base.Callable, Nothing}=nothing,
)::Modelset
    # if model is unspecified, use default model setup
    isnothing(model) && (model = DEFAULT_MODEL_SETUP)
    modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, rules, preprocess, reducefunc)
    Modelset(modelset, _prepare_dataset(X, y, modelset))
end

# y is not a vector, but a symbol or a string that identifies a column in X
function prepare_dataset(
    X::AbstractDataFrame,
    y::Union{Symbol,AbstractString};
    kwargs...
)::Modelset
    prepare_dataset(X[!, Not(y)], X[!, y]; kwargs...)
end
