# this module transforms multidimensional datasets (tested only on time-series)
# into formats suitable for different model algorithm families:

# 1. propositional algorithms (DecisionTree, XGBoost):
#    - applies windowing to divide time series into segments
#    - extracts scalar features (max, min, mean, etc.) from each window
#    - returns a standard tabular DataFrame

# 2. modal algorithms (ModalDecisionTree):
#    - creates windowed time series preserving temporal structure
#    - applies reduction functions to manage dimensionality

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
# base type for metadata containers
abstract type AbstractTreatmentInfo end

"""
    AbstractWinFunction

Base type for windowing function implementations.
"""
abstract type AbstractWinFunction end

# ---------------------------------------------------------------------------- #
#                                  windowing                                   #
# ---------------------------------------------------------------------------- #
"""
    WinFunction <: AbstractWinFunction

Callable wrapper for windowing algorithms with parameters.

# Fields
- `func::Function`: The windowing implementation function
- `params::NamedTuple`: Algorithm-specific parameters
"""
struct WinFunction <: AbstractWinFunction
    func   :: Function
    params :: NamedTuple
end
# make it callable - npoints is passed at execution time
(w::WinFunction)(npoints::Int; kwargs...) = w.func(npoints; w.params..., kwargs...)

"""
    MovingWindow(; window_size::Int, window_step::Int) -> WinFunction

Create a moving window that slides across the time series.

# Parameters
- `window_size`: Number of time points in each window
- `window_step`: Step size between consecutive windows

# Example
```
win = MovingWindow(window_size=10, window_step=5)
intervals = win(100)  # For 100-point time series
```
"""
function MovingWindow(;
    window_size::Int,
    window_step::Int,
)
    WinFunction(movingwindow, (;window_size, window_step))
end

"""
    WholeWindow() -> WinFunction

Create a single window encompassing the entire time series.
Useful for global feature extraction without temporal partitioning.

# Example
```
win = WholeWindow()
intervals = win(100)  # Returns [1:100]
```
"""
WholeWindow(;) = WinFunction(wholewindow, (;))

"""
    SplitWindow(; nwindows::Int) -> WinFunction

Divide the time series into equal non-overlapping segments.

# Parameters
- `nwindows`: Number of equal-sized windows to create

# Example
```
win = SplitWindow(nwindows=4)
intervals = win(100)  # Four 25-point windows
```
"""
SplitWindow(;nwindows::Int) = WinFunction(splitwindow, (; nwindows))

"""
    AdaptiveWindow(; nwindows::Int, relative_overlap::AbstractFloat) -> WinFunction

Create overlapping windows with adaptive sizing based on series length.

# Parameters
- `nwindows`: Target number of windows
- `relative_overlap`: Fraction of overlap between adjacent windows (0.0-1.0)

# Example
```
win = AdaptiveWindow(nwindows=3, relative_overlap=0.1)
intervals = win(100)  # Three adaptive windows with 10% overlap
```
"""
function AdaptiveWindow(;
    nwindows::Int,
    relative_overlap::AbstractFloat,
)
    WinFunction(adaptivewindow, (; nwindows, relative_overlap))
end

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
# apply a feature reduce function to all time-series in a column
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{<:Vector{<:Real}},
    feature_func::Function,
    col_name::Symbol
)::Vector{<:Real}
    @views @inbounds X[!, col_name] = collect(feature_func(col) for col in X_col)
end

# apply a feature function to a specific time interval within each time-series
# - interval: time range to extract features from (e.g., 1:50 for first 50 points)
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{<:Vector{<:Real}},
    feature_func::Function,
    col_name::Symbol,
    interval::UnitRange{Int64}
)::Vector{<:Real}
    @views @inbounds X[!, col_name] = collect(feature_func(col[interval]) for col in X_col)
end

# apply a reduction function across multiple intervals for modal algorithm preparation
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{<:Vector{<:Real}},
    modalreduce_func::Function,
    col_name::Symbol,
    intervals::Vector{UnitRange{Int64}}
)::Vector{<:Vector{<:Real}}
    X[!, col_name] = [
        [modalreduce_func(@view(ts[interval])) for interval in intervals]
        for ts in X_col
    ]
end

# check dataframe
is_multidim_dataframe(X::DataFrame)::Bool =
    any(eltype(col) <: AbstractArray for col in eachcol(X))

# ---------------------------------------------------------------------------- #
#                          multidimensional dataset                            #
# ---------------------------------------------------------------------------- #
# metadata container for dataset preprocessing operations.
struct TreatmentInfo <: AbstractTreatmentInfo
    features    :: Tuple{Vararg{Base.Callable}}
    winparams   :: WinFunction
    treatment   :: Symbol
    modalreduce :: Base.Callable
end

# simplified metadata for aggregation-only preprocessing.
struct AggregationInfo <: AbstractTreatmentInfo
    features    :: Tuple{Vararg{Base.Callable}}
    winparams   :: WinFunction
end

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
get_treatment(t::TreatmentInfo) = t.treatment

# ---------------------------------------------------------------------------- #
#                                   base show                                  #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, info::TreatmentInfo)
    println(io, "TreatmentInfo:")
    for field in fieldnames(TreatmentInfo)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 15), value)
    end
end

function Base.show(io::IO, info::AggregationInfo)
    println(io, "AggregationInfo:")
    for field in fieldnames(AggregationInfo)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 15), value)
    end
end

# ---------------------------------------------------------------------------- #
#                        modal -> propositional adapter                        #
# ---------------------------------------------------------------------------- #
# convert treatment information (features and winparams) to aggregation information.
treat2aggr(t::TreatmentInfo)::AggregationInfo = 
    AggregationInfo(t.features, t.winparams)

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function treatment end

"""
    treatment(X::AbstractDataFrame; treat::Symbol, win::WinFunction, 
             features::Tuple, modalreduce::Base.Callable) -> (DataFrame, TreatmentInfo)

Transform multidimensional dataset based on specified treatment strategy.

# Arguments
- `X::AbstractDataFrame`: Input dataset with time series in each cell
- `treat::Symbol`: Treatment type - :aggregate, :reducesize or :none
- `win::WinFunction`: Windowing strategy (default: `AdaptiveWindow(nwindows=3, relative_overlap=0.1)`)
- `features::Tuple`: Feature extraction functions (default: `(maximum, minimum)`)
- `modalreduce::Base.Callable`: Reduction function for modal treatments (default: `mean`)

# Treatment Types

## `:aggregate` (for Propositional Algorithms)
Extracts scalar features from time series windows:
- Single window: Applies features to entire time series
- Multiple windows: Creates feature columns per window (e.g., "max(col1)w1")

## `:reducesize` (for Modal Algorithms)
Preserves temporal structure while reducing dimensionality:
- Applies reduction function to each window
- Maintains Vector{Float64} format for modal logic compatibility

## `:none` (for particular cases)
Returns the dataset
"""
function treatment(
    X           :: AbstractDataFrame,
    treat       :: Symbol;
    win         :: WinFunction=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features    :: Tuple{Vararg{Base.Callable}}=(maximum, minimum),
    modalreduce :: Base.Callable=mean
)
    is_multidim_dataframe(X) || throw(ArgumentError("Input DataFrame " * 
        "does not contain multidimensional data."))

    vnames = propertynames(X)
    _X = DataFrame()
    intervals = win(length(X[1,1]))

    # propositional models
    isempty(features) && (treat = :none)

    if treat == :aggregate
        for f in features, v in vnames
            if length(intervals) == 1
                # single window: apply to whole time series
                col_name = Symbol("$(f)($(v))")
                apply_vectorized!(_X, X[!, v], f, col_name)
            else
                # multiple windows: apply to each interval
                for (i, interval) in enumerate(intervals)
                    col_name = Symbol("$(f)($(v))w$(i)")
                    apply_vectorized!(_X, X[!, v], f, col_name, interval)
                end
            end
        end

    # modal models
    elseif treat == :reducesize
        for v in vnames
            apply_vectorized!(_X, X[!, v], modalreduce, v, intervals)
        end
        
    elseif treat == :none
        _X = X

    else
        error("Unknown treatment type: $treat")
    end

    return _X, TreatmentInfo(features, win, treat, modalreduce)
end