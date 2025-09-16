"""
treatment.jl — Multidimensional Dataset Preprocessing

This module transforms multidimensional datasets (especially time series) into formats
suitable for different algorithm families:

1. Propositional algorithms (DecisionTree, XGBoost):
   - Applies windowing to divide time series into segments
   - Extracts scalar features (max, min, mean, etc.) from each window
   - Returns a standard tabular DataFrame

2. Modal algorithms (ModalDecisionTree):
   - Creates windowed time series preserving temporal structure
   - Applies reduction functions to manage dimensionality

Key components:
- WinFunction: Configurable windowing strategies (moving, adaptive, split)
- TreatmentInfo: Metadata about applied transformations
- treatment(): Main preprocessing interface

Currently supports numeric and time series data with plans for arbitrary
dimensional extensions.
"""

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractTreatmentInfo

Base type for metadata containers.
"""
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
# Make it callable - npoints is passed at execution time
(w::WinFunction)(npoints::Int; kwargs...) = w.func(npoints; w.params..., kwargs...)

"""
    MovingWindow(; window_size::Int, window_step::Int) -> WinFunction

Create a moving window that slides across the time series.

# Parameters
- `window_size`: Number of time points in each window
- `window_step`: Step size between consecutive windows

# Example
    win = MovingWindow(window_size=10, window_step=5)
    intervals = win(100)  # For 100-point time series
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
    win = WholeWindow()
    intervals = win(100)  # Returns [1:100]
"""
WholeWindow(;) = WinFunction(wholewindow, (;))

"""
    SplitWindow(; nwindows::Int) -> WinFunction

Divide the time series into equal non-overlapping segments.

# Parameters
- `nwindows`: Number of equal-sized windows to create

# Example
    win = SplitWindow(nwindows=4)
    intervals = win(100)  # Four 25-point windows
"""
SplitWindow(;nwindows::Int) = WinFunction(splitwindow, (; nwindows))

"""
    AdaptiveWindow(; nwindows::Int, relative_overlap::AbstractFloat) -> WinFunction

Create overlapping windows with adaptive sizing based on series length.

# Parameters
- `nwindows`: Target number of windows
- `relative_overlap`: Fraction of overlap between adjacent windows (0.0-1.0)

# Example
    win = AdaptiveWindow(nwindows=3, relative_overlap=0.1)
    intervals = win(100)  # Three adaptive windows with 10% overlap
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
"""
    apply_vectorized!(X::DataFrame, X_col::Vector{Vector{Float64}}, 
                     feature_func::Function, col_name::Symbol) -> Nothing

Apply a feature extraction function to all time series in a column.

# Arguments
- `X`: Target DataFrame to receive new feature column
- `X_col`: Vector of time series (each element is a Vector{Float64})
- `feature_func`: Function to apply to each time series (e.g., maximum, minimum)
- `col_name`: Name for the new feature column
"""
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{Vector{Float64}},
    feature_func::Function,
    col_name::Symbol
)::Nothing
    X[!, col_name] = collect(feature_func(X_col[i]) for i in 1:length(X_col))
    return nothing
end

"""
    apply_vectorized!(X::DataFrame, X_col::Vector{Vector{Float64}}, 
                     feature_func::Function, col_name::Symbol, 
                     interval::UnitRange{Int64}) -> Nothing

Apply a feature function to a specific time interval within each time series.

# Additional Arguments
- `interval`: Time range to extract features from (e.g., 1:50 for first 50 points)

# Example
    apply_vectorized!(df, ts_column, mean, :window1_mean, 1:25)
"""
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{Vector{Float64}},
    feature_func::Function,
    col_name::Symbol,
    interval::UnitRange{Int64}
)::Nothing
    X[!, col_name] = collect(feature_func(@views X_col[i][interval]) for i in 1:length(X_col))
    return nothing
end

"""
    apply_vectorized!(X::DataFrame, X_col::Vector{Vector{Float64}}, 
                     modalreduce_func::Function, col_name::Symbol,
                     intervals::Vector{UnitRange{Int64}}, n_rows::Int, 
                     n_intervals::Int) -> Nothing

Apply a reduction function across multiple intervals for modal algorithm preparation.

# Arguments
- `modalreduce_func`: Reduction function applied to each interval
- `intervals`: Vector of time ranges defining windows
- `n_rows`: Number of time series in the dataset
- `n_intervals`: Number of intervals per time series
"""
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{Vector{Float64}},
    modalreduce_func::Function,
    col_name::Symbol,
    intervals::Vector{UnitRange{Int64}},
    n_rows::Int,
    n_intervals::Int
)::Nothing
    result_column = Vector{Vector{Float64}}(undef, n_rows)
    row_result = Vector{Float64}(undef, n_intervals)
    
    @inbounds @fastmath for row_idx in 1:n_rows
        ts = X_col[row_idx]
        
        for (i, interval) in enumerate(intervals)
            row_result[i] = modalreduce_func(@view(ts[interval]))
        end
        result_column[row_idx] = copy(row_result)
    end

    X[!, col_name] = result_column
    
    return nothing
end

# ---------------------------------------------------------------------------- #
#                          multidimensional dataset                            #
# ---------------------------------------------------------------------------- #
"""
    TreatmentInfo <: AbstractTreatmentInfo

Metadata container for dataset preprocessing operations.

# Fields
- `features::Tuple{Vararg{Base.Callable}}`: Feature extraction functions applied
- `winparams::WinFunction`: Windowing strategy used
- `treatment::Symbol`: Treatment type (:aggregate, :reducesize, :none)
- `modalreduce::Base.Callable`: Reduction function for modal treatments
"""
struct TreatmentInfo <: AbstractTreatmentInfo
    features    :: Tuple{Vararg{Base.Callable}}
    winparams   :: WinFunction
    treatment   :: Symbol
    modalreduce :: Base.Callable
end

"""
    AggregationInfo <: AbstractTreatmentInfo

Simplified metadata for aggregation-only preprocessing.

# Fields
- `features::Tuple{Vararg{Base.Callable}}`: Feature functions used
- `winparams::WinFunction`: Windowing configuration
"""
struct AggregationInfo <: AbstractTreatmentInfo
    features    :: Tuple{Vararg{Base.Callable}}
    winparams   :: WinFunction
end

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
- `treat::Symbol`: Treatment type - :aggregate, :reducesize, or :none
- `win::WinFunction`: Windowing strategy (default: AdaptiveWindow(nwindows=3, relative_overlap=0.1))
- `features::Tuple`: Feature extraction functions (default: (maximum, minimum))
- `modalreduce::Base.Callable`: Reduction function for modal treatments (default: mean)

# Treatment Types

## :aggregate (Propositional Algorithms)
Extracts scalar features from time series windows:
- Single window: Applies features to entire time series
- Multiple windows: Creates feature columns per window (e.g., "max(col1)w1")

## :reducesize (Modal Algorithms)
Preserves temporal structure while reducing dimensionality:
- Applies reduction function to each window
- Maintains Vector{Float64} format for modal logic compatibility
"""
function treatment(
    X           :: AbstractDataFrame;
    treat       :: Symbol,
    win         :: WinFunction=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features    :: Tuple{Vararg{Base.Callable}}=(maximum, minimum),
    modalreduce :: Base.Callable=mean
)
    vnames = propertynames(X)
    n_rows = nrow(X)
    _X = DataFrame()

    # run the windowing algo and set windows indexes
    intervals = win(length(X[1,1]))
    n_intervals = length(intervals)

    # define column names and prepare data structure based on treatment type
    # propositional
    isempty(features) && (treat = :none)

    if treat == :aggregate
        if n_intervals == 1
            # apply feature to whole time-series
            for f in features
                @simd for v in vnames
                    col_name = Symbol(f, "(", v, ")")
                    apply_vectorized!(_X, X[!, v], f, col_name)
                end
            end
        else
            # apply feature to specific intervals
            for f in features
                @simd for v in vnames
                    for (i, interval) in enumerate(intervals)
                        col_name = Symbol(f, "(", v, ")w", i)
                        apply_vectorized!(_X, X[!, v], f, col_name, interval)
                    end
                end
            end
        end

    # modal
    elseif treat == :reducesize
        for v in vnames
            apply_vectorized!(_X, X[!, v], modalreduce, v, intervals, n_rows, n_intervals)
        end
    
    elseif treat == :none
        _X = X
    else
        error("Unknown treatment type: $treat")
    end

    return _X, TreatmentInfo(features, win, treat, modalreduce)
end