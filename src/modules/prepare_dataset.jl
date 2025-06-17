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
        ref_idx === nothing && continue
        
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
            X_coded = MLJ.levelcode.(categorical(col)) 
            X[!, name] = X_coded
        end
    end
    
    return X
end

function code_dataset(y::AbstractVector)
    if !(eltype(y) <: Number)
        eltype(y) <: Symbol && (y = string.(y))
        y = MLJ.levelcode.(categorical(y)) 
    end
    
    return y
end

code_dataset(X::AbstractDataFrame, y::AbstractVector) = code_dataset(X), code_dataset(y)

"""
    check_dimensions(X::AbstractMatrix) -> Int

Check that all elements in a matrix have consistent dimensions.

# Arguments
- `X::AbstractMatrix`: Matrix containing array-like elements to check for dimension consistency

# Returns
- `Int`: The number of dimensions of the elements (0 if matrix is empty)

# Throws
- `ArgumentError`: If elements have more than 1 dimension
- `DimensionMismatch`: If elements have inconsistent dimensions
"""
function check_dimensions(X::AbstractMatrix)
    isempty(X) && return 0
    
    # Get reference dimensions from first element
    first_col = first(eachcol(X))
    ref_dims = ndims(first(first_col))
    
    # Early dimension check
    ref_dims > 1 && throw(ArgumentError("Elements more than 1D are not supported."))
    
    # Check all columns maintain same dimensionality
    all(col -> all(x -> ndims(x) == ref_dims, col), eachcol(X)) ||
        throw(DimensionMismatch("Inconsistent dimensions across elements"))
    
    return ref_dims
end

check_dimensions(df::DataFrame) = check_dimensions(Matrix(df))

"""
    find_max_length(X::AbstractMatrix) -> Tuple{Vararg{Int}}

Find the maximum dimensions of elements in a matrix containing either scalar values or array-like elements.

# Arguments
- `X::AbstractMatrix`: A matrix where each element can be either a scalar or an array-like structure

# Returns
- `Tuple{Vararg{Int}}`: A tuple containing the maximum sizes:
  - For empty matrices: Returns `0`
  - For matrices with scalar values: Returns `(1,)`
  - For matrices with vector elements: Returns `(max_length,)` where `max_length` is the length of the longest vector
  - For matrices with multi-dimensional arrays: Returns a tuple with maximum size in each dimension
"""
function find_max_length(X::AbstractMatrix)
    isempty(X) && return 0
    
    # check the type of the first element to determine DataFrame structure
    first_element = first(skipmissing(first(eachcol(X))))
    
    if first_element isa Number
        return (1,)
    else
        ndims_val = ndims(first_element)
        # for each dimension, find the maximum size
        ntuple(ndims_val) do dim
            mapreduce(col -> maximum(x -> size(x, dim), col), max, eachcol(X); init=0)
        end
    end
end

find_max_length(df::DataFrame) = find_max_length(Matrix(df))

# ---------------------------------------------------------------------------- #
#                                 treatment                                    #
# ---------------------------------------------------------------------------- #
"""
    _treatment(X::AbstractMatrix{T}, vnames::VarNames, treatment::Symbol,
              features::FeatNames, winparams::WinParams; 
              modalreduce::Base.Callable=mean) -> Tuple{Matrix, Vector{String}}

Process a matrix data by applying feature extraction or dimension reduction.

# Arguments
- `X::AbstractMatrix{T}`: Matrix where each element is a time series (array) or scalar value
- `vnames::VarNames`: Names of variables/columns in the original data
- `treatment::Symbol`: Treatment method to apply:
  - `:aggregate`: Extract features from time series (propositional approach)
  - `:reducesize`: Reduce time series dimensions while preserving temporal structure
- `features::FeatNames`: Functions to extract features from time series segments
- `winparams::WinParams`: Parameters for windowing time series:
  - `type`: Window function to use (e.g., `adaptivewindow`, `wholewindow`)
  - `params`: Additional parameters for the window function
- `modalreduce::Base.Callable=mean`: Function to reduce windows in `:reducesize` mode (default: `mean`)

# Returns
- `Tuple{Matrix, Vector{String}}`: Processed matrix and column names:
  - For `:aggregate`: Matrix of extracted features with column names like `"func(var)w1"`
  - For `:reducesize`: Matrix where each cell contains a reduced vector with original column names

# Details
## Aggregate Treatment
When `treatment = :aggregate`:
1. Divides each time series into windows using the specified windowing function
2. Applies each feature function to each window of each variable
3. Creates a feature matrix where each row contains features extracted from original data
4. Handles variable-length time series by padding with NaN values as needed
5. Column names include function name, variable name and window index (e.g. "mean(temp)w1")

## Reducesize Treatment
When `treatment = :reducesize`:
1. Divides each time series into windows using the specified windowing function
2. Applies the reduction function to each window (by default `mean`)
3. Returns a matrix where each element is a reduced-length vector
4. Maintains original column names
"""
function _treatment(
    X::AbstractMatrix{T},
    vnames::VarNames,
    treatment::Symbol,
    features::Union{Vector{<:Base.Callable}, Nothing},
    winparams::WinParams;
    modalreduce::OptCallable=nothing
) where T
    # working with audio files, we need to consider audio of different lengths.
    max_interval = first(find_max_length(X))
    n_intervals = winparams.type(max_interval; winparams.params...)

    # define column names and prepare data structure based on treatment type
    if treatment == :aggregate        # propositional
        if n_intervals == 1
            col_names = [string(f, "(", v, ")") for f in features for v in vnames]
            
            n_rows = size(X, 1)
            n_cols = length(col_names)
            result_matrix = Matrix{eltype(T)}(undef, n_rows, n_cols)
        else
            # define column names with features names and window indices
            col_names = [string(f, "(", v, ")w", i) 
                         for f in features 
                         for v in vnames 
                         for i in 1:length(n_intervals)]
            
            n_rows = size(X, 1)
            n_cols = length(col_names)
            result_matrix = Matrix{eltype(T)}(undef, n_rows, n_cols)
        end
            
        # fill matrix
        for (row_idx, row) in enumerate(eachrow(X))
            row_intervals = winparams.type(maximum(length.(collect(row))); winparams.params...)
            interval_diff = length(n_intervals) - length(row_intervals)

            # calculate feature values for this row
            feature_values = vcat([
                vcat([f(col[r]) for r in row_intervals],
                    fill(NaN, interval_diff)) for col in row, f in features
            ]...)
            result_matrix[row_idx, :] = feature_values
        end

    elseif treatment == :reducesize   # modal
        @show "Q"
        col_names = vnames
        
        n_rows = size(X, 1)
        n_cols = length(col_names)
        result_matrix = Matrix{T}(undef, n_rows, n_cols)

        modalreduce === nothing && (modalreduce = mean)
        
        for (row_idx, row) in enumerate(eachrow(X))
            row_intervals = winparams.type(maximum(length.(collect(row))); winparams.params...)
            interval_diff = length(n_intervals) - length(row_intervals)
            
            # calculate reduced values for this row
            reduced_data = [
                vcat([modalreduce(col[r]) for r in row_intervals],
                     fill(NaN, interval_diff)) for col in row
            ]
            result_matrix[row_idx, :] = reduced_data
        end
    end

    return result_matrix, col_names
end

# _treatment(df::DataFrame, args...; kwargs...) = _treatment(Matrix(df), args...; kwargs...)

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
    resample::Resample,
    rng::AbstractRNG
)::Union{TT_indexes{Int}, Vector{TT_indexes{Int}}}
    # if resample === nothing
    #     tt = MLJ.partition(eachindex(y), train_ratio; shuffle=true, rng)
    #     if valid_ratio == 1.0
    #         return TT_indexes(tt[1], eltype(tt[1])[], tt[2])
    #     else
    #         tv = MLJ.partition(tt[1], valid_ratio; shuffle=true, rng)
    #         return TT_indexes(tv[1], tv[2], tt[2])
    #     end
    # else
        resample_cv = resample.type(; resample.params...)
        tt = MLJ.MLJBase.train_test_pairs(resample_cv, 1:length(y), y)
        if valid_ratio == 1.0
            return [TT_indexes(train, eltype(train)[], test) for (train, test) in tt]
        else
            tv = collect((MLJ.partition(t[1], train_ratio)..., t[2]) for t in tt)
            return [TT_indexes(train, valid, test) for (train, valid, test) in tv]
        end
    # end
end

# ---------------------------------------------------------------------------- #
#                               prepare dataset                                #
# ---------------------------------------------------------------------------- #
"""
    prepare_dataset(X::AbstractDataFrame, y::AbstractVector; kwargs...)::Modelset
    prepare_dataset(X::AbstractDataFrame, y::SymbolString; kwargs...)::Modelset

Prepares a dataset for machine learning by processing the input data and configuring a model setup.
Supports both classification and regression tasks, with extensive customization options.

# Arguments
- `X::AbstractDataFrame`: The input data containing features
- `y::AbstractVector` or `y::SymbolString`: The target variable, either as a vector or 
  as a column name/symbol from `X`

# Optional Keyword Arguments
- `model::OptNamedTuple=nothing`: Model configuration with fields:
  - `type`: Model type (e.g., `:xgboost`, `:randomforest`)
  - `params`: Model-specific parameters
- `resample::OptNamedTuple=nothing`: Resampling strategy
  - `type`: Resampling method (e.g., `:cv`, `:stratifiedcv`)
  - `params`: Resampling parameters like `nfolds`
- `win::OptNamedTuple=nothing`: Windowing parameters for time series data
  - `type`: Window function (e.g., `adaptivewindow`, `wholewindow`)
  - `params`: Window parameters like `nwindows`
- `features::OptTuple=nothing`: Statistical functions to extract from time series
  (e.g., `(mean, std, maximum)`)
- `tuning::OptNamedTupleBool=nothing`: Hyperparameter tuning configuration
- `rules::OptNamedTuple=nothing`: Rules for post-hoc explanation
- `preprocess::OptNamedTuple=nothing`: Data preprocessing parameters:
  - `train_ratio`: Ratio of data for training vs testing
  - `valid_ratio`: Ratio of training data for validation
  - `rng`: Random number generator
- `modalreduce::OptCallable=nothing`: Function for reducing time series data
  in `:reducesize` treatment mode (default: `mean`)

# Returns
- `Modelset`: A complete modelset containing both the configured model setup and prepared dataset

# Notes
- For non-numeric data in `X`, use `code_dataset(X)` before calling this function
- Time series data should be stored as vectors within DataFrame cells
- If `y` is provided as a column name, it will be extracted from `X`
- When `model=nothing`, a default model setup is used
"""
function __prepare_dataset(
    df::AbstractDataFrame,
    y::AbstractVector;
    algo::DataType,
    treatment::Symbol,
    features::Vector{<:Base.Callable},
    train_ratio::Float64,
    valid_ratio::Float64,
    rng::AbstractRNG,
    resample::Union{Resample, Nothing},
    winparams::WinParams,
    vnames::Union{VarNames,Nothing}=nothing,
    modalreduce::OptCallable=nothing
)::Dataset
    X = Matrix(df)
    # check parameters
    check_dataset_type(X) || throw(ArgumentError("DataFrame must contain only numeric values, use SoleXplorer.code_dataset() to convert non-numeric data"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))
    check_row_consistency(X) || throw(ArgumentError("Elements within each row must have consistent dimensions"))
    # treatment in AVAIL_TREATMENTS || throw(ArgumentError("Treatment must be one of: $AVAIL_TREATMENTS"))

    if algo == AbstractRegression
        y isa AbstractVector{<:Reg_Value} || throw(ArgumentError("Regression requires a numeric target variable"))
        y isa AbstractFloat || (y = Float64.(y))
    elseif algo == AbstractClassification
        y isa AbstractVector{<:Cat_Value} || throw(ArgumentError("Classification requires a categorical target variable"))
        y isa MLJ.CategoricalArray || (y = coerce(y, MLJ.Multiclass))
    end

    if vnames === nothing
        vnames = names(df)
    else
        size(X, 2) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
        vnames isa AbstractVector{<:AbstractString} || (vnames = string.(vnames))
    end

    hasnans(X) && @warn "DataFrame contains NaN values"

    column_eltypes = eltype.(eachcol(X))

    if all(t -> t <: AbstractVector{<:Number}, column_eltypes) && !(winparams === nothing)
        X, vnames = _treatment(X, vnames, treatment, [features...], winparams; modalreduce)
    end

    ds_info = DatasetInfo(
        treatment,
        modalreduce,
        train_ratio,
        valid_ratio,
        rng,
        vnames
    )

    return Dataset(
        X, y,
        _partition(y, train_ratio, valid_ratio, resample, rng),
        ds_info
    )
end

function __prepare_dataset(
    X::AbstractDataFrame,
    y::AbstractVector,
    model::AbstractModelSetup
)::Dataset
    # modal reduce function, optional for propositional
    # modalreduce = haskey(model.preprocess, :modalreduce) ? model.config.modalreduce : nothing

    __prepare_dataset(
        X, y;
        algo=modeltype(model),
        treatment=model.config.treatment,
        features=model.features,
        train_ratio=model.preprocess.train_ratio,
        valid_ratio=model.preprocess.valid_ratio,
        rng=model.preprocess.rng,
        resample=model.resample,
        winparams=model.winparams,
        vnames=model.preprocess.vnames,
        modalreduce=model.preprocess.modalreduce,
    )
end

function _prepare_dataset(
    X             :: AbstractDataFrame,
    y             :: AbstractVector;
    model         :: NamedTuple     = (;type=:decisiontree),
    resample      :: NamedTuple     = (;type=Holdout),
    win           :: OptNamedTuple  = nothing,
    features      :: OptTuple       = nothing,
    tuning        :: NamedTupleBool = false,
    extract_rules :: NamedTupleBool = false,
    preprocess    :: OptNamedTuple  = nothing,
    # modalreduce    :: OptCallable    = nothing,
    measures      :: OptTuple       = nothing,
)::Tuple{Modelset, Dataset}
    modelset = validate_modelset(
        model, eltype(y);
        resample,
        win,
        features,
        tuning,
        extract_rules,
        preprocess,
        # modalreduce,
        measures
    )
    Modelset(modelset,), __prepare_dataset(X, y, modelset)
end

prepare_dataset(args...; kwargs...)::Tuple{Modelset, Dataset} = _prepare_dataset(args...; kwargs...)

# y is not a vector, but a symbol or a string that identifies a column in X
function prepare_dataset(
    X::AbstractDataFrame,
    y::SymbolString;
    kwargs...
)::Tuple{Modelset, Dataset}
    prepare_dataset(X[!, Not(y)], X[!, y]; kwargs...)
end
