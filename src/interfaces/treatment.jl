# questa struttura conserva tutte le informazioni necessarie per trattare un dataset multidimensionale.
# per ora, Sole accetta solo dataset numerici e, al più, serie temporali;
# ma è nostra intenzione estenderlo a qualsiasi dimensione di dataset.
# il fine ultimo è: 
# - per l'utilizzo dei normali algoritmi proposizionali, quali decision tree e xgboost,
# convertire il dataset multidimensionale, in un dataset numerico, suddividendo le serie temporali in 
# finestre, alle quali applicare una condizione.
# - per l'utilizzo di algoritmi modali, quali modaldecisiontree, 
# viene creata una struttura ad hoc: sole logiset.
# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractTreatmentInfo end
abstract type AbstractWinFunction end

# ---------------------------------------------------------------------------- #
#                                  windowing                                   #
# ---------------------------------------------------------------------------- #
struct WinFunction <: AbstractWinFunction
    func   :: Function
    params :: NamedTuple
end
# Make it callable - npoints is passed at execution time
(w::WinFunction)(npoints::Int; kwargs...) = w.func(npoints; w.params..., kwargs...)

function MovingWindow(;
    window_size::Int,
    window_step::Int,
)
    WinFunction(movingwindow, (;window_size, window_step))
end

WholeWindow(;) = WinFunction(wholewindow, (;))
SplitWindow(;nwindows::Int) = WinFunction(splitwindow, (; nwindows))

function AdaptiveWindow(;
    nwindows::Int,
    relative_overlap::AbstractFloat,
)
    WinFunction(adaptivewindow, (; nwindows, relative_overlap))
end

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{Vector{Float64}},
    feature_func::Function,
    col_name::Symbol
)::Nothing
    X[!, col_name] = collect(feature_func(X_col[i]) for i in 1:length(X_col))
    return nothing
end

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
mutable struct TreatmentInfo <: AbstractTreatmentInfo
    features    :: Vector{<:Base.Callable}
    winparams   :: WinFunction
    treatment   :: Symbol
    modalreduce :: Base.Callable
end

function Base.show(io::IO, info::TreatmentInfo)
    println(io, "TreatmentInfo:")
    for field in fieldnames(TreatmentInfo)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 15), value)
    end
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function treatment end

function treatment(
    X           :: AbstractDataFrame;
    treat       :: Symbol,
    win         :: WinFunction=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features    :: Vector{<:Base.Callable}=[maximum, minimum],
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

    else
        error("Unknown treatment type: $treat")
    end

    return _X, TreatmentInfo(features, win, treat, modalreduce)
end