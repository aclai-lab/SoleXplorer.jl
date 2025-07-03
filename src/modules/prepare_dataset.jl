# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
check_dataset_type(X::AbstractDataFrame) = eltype(X) <: Union{Real,AbstractArray{<:Real}}
check_dataset_type(X::AbstractMatrix)    = eltype(X) <: Union{Real,AbstractArray{<:Real}}

hasnans(X::AbstractDataFrame) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))
# hasnans(X::AbstractMatrix) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))

# conversion of variable names to symbols
symbols_generator(strings::AbstractVector{<:AbstractString})::Base.Generator{Vector{String}} = (Symbol(s) for s in strings)
symbols_generator(strings::MLJ.CategoricalArray)::Base.Generator{<:MLJ.CategoricalVector}    = (Symbol(s) for s in strings)

# check time series consistencies
function check_row_consistency(X::AbstractDataFrame) 
    for row in eachrow(X)
        # skip cols with only scalar values
        any(el -> el isa AbstractArray, row) || continue
        
        # find first array element to use as reference
        ref_idx = findfirst(el -> el isa AbstractArray, row)
        ref_idx === nothing && continue
        
        ref_size = size(row[ref_idx])
        
        # check if any array element has different size (short-circuit)
        if any(row) do el
                el isa AbstractArray && size(el) != ref_size
            end
            throw(DimensionMismatch("Inconsistent dimensions across elements in row: $(row)"))
        end
    end
end

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
function _treatment(
    X::AbstractMatrix{T},
    vnames::Vector{Symbol},
    treatment::Symbol,
    features::Union{Vector{<:Base.Callable}, Nothing},
    winparams::WinParams;
    modalreduce::Base.Callable=mean
) where T
    # working with audio files, we need to consider audio of different lengths.
    max_interval = first(find_max_length(X))
    n_intervals = winparams.type(max_interval; winparams.params...)

    # define column names and prepare data structure based on treatment type
    if treatment == :aggregate        # propositional
        if length(n_intervals) == 1
            col_names = [(f, "(", v, ")") for f in features for v in vnames]
            
            n_rows = size(X, 1)
            n_cols = length(col_names)
            result_matrix = Matrix{eltype(T)}(undef, n_rows, n_cols)
        else
            # define column names with features names and window indices
            col_names = [(f, "(", v, ")w", i) 
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

# ---------------------------------------------------------------------------- #
#                                 partitioning                                 #
# ---------------------------------------------------------------------------- #
function _partition(
    y::AbstractVector{<:SoleModels.Label},
    train_ratio::Real,
    valid_ratio::Real,
    resample::Resample
)::Union{DataSplit{Int}, Vector{DataSplit{Int}}}
    resample_cv = resample.type(; resample.params...)
    tt = MLJ.MLJBase.train_test_pairs(resample_cv, 1:length(y), y)
    if valid_ratio == 1.0
        return [DataSplit(train, eltype(train)[], test) for (train, test) in tt]
    else
        tv = collect((MLJ.partition(t[1], train_ratio)..., t[2]) for t in tt)
        return [DataSplit(train, valid, test) for (train, valid, test) in tv]
    end
end

# ---------------------------------------------------------------------------- #
#                               prepare dataset                                #
# ---------------------------------------------------------------------------- #
function __prepare_dataset(
    X     :: AbstractDataFrame,
    y     :: AbstractVector,
    model :: AbstractModelSetup
) :: Dataset
    __prepare_dataset(
        X, y;
        dataset_args(model)...
    )
end

function __prepare_dataset(
    X           :: AbstractDataFrame,
    y           :: AbstractVector{<:SoleModels.Label};
    algo        :: DataType,
    treatment   :: Symbol,
    features    :: Vector{<:Base.Callable},
    train_ratio :: Real,
    valid_ratio :: Real,
    rng         :: AbstractRNG,
    resample    :: Union{Resample, Nothing},
    winparams   :: WinParams,
    vnames      :: Union{VarNames,Nothing}=nothing,
    modalreduce :: OptCallable=nothing
) :: Dataset
    # X = Matrix(df)

    # check parameters
    # check_dataset_type(X) || throw(ArgumentError("DataFrame must contain only numeric values, use SoleXplorer.code_dataset() to convert non-numeric data"))
    nrow(X) == length(y)     || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))
    check_row_consistency(X)

    if algo == AbstractRegression
        y isa AbstractVector{<:SoleModels.RLabel} || throw(ArgumentError("Regression requires a numeric target variable"))
        y isa AbstractFloat || (y = Float64.(y))
    elseif algo == AbstractClassification
        y isa AbstractVector{<:SoleModels.CLabel} || throw(ArgumentError("Classification requires a categorical target variable"))
        y isa Vector{Symbol} || (y = collect(symbols_generator(y)))
    end

    if vnames === nothing
        vnames = propertynames(X)
    else
        ncol(X) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
        vnames isa Vector{Symbol} || (vnames = collect(symbols_generator(vnames)))
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
        _partition(y, train_ratio, valid_ratio, resample),
        ds_info
    )
end

dataset_args(model::AbstractModelSetup)::NamedTuple = (
    algo        = modeltype(       model),
    treatment   = get_treatment(   model),
    features    = get_features(    model),
    train_ratio = get_train_ratio( model),
    valid_ratio = get_valid_ratio( model),
    rng         = get_rng(         model),
    resample    = get_resample(    model),
    winparams   = get_winparams(   model),
    vnames      = get_vnames(      model),
    modalreduce = get_modalreduce( model)
)

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
        measures
    )
    Modelset(modelset,), __prepare_dataset(X, y, modelset)
end

prepare_dataset(args...; kwargs...)::Tuple{Modelset, Dataset} = _prepare_dataset(args...; kwargs...)

# y is not a vector, but a symbol or a string that identifies a column in X
function prepare_dataset(
    X :: AbstractDataFrame,
    y :: SymbolString;
    kwargs...
) :: Tuple{Modelset, Dataset}
    prepare_dataset(X[!, Not(y)], X[!, y]; kwargs...)
end
