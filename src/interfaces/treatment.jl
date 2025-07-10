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
#                          multidimensional dataset                            #
# ---------------------------------------------------------------------------- #
mutable struct TreatmentInfo <: AbstractTreatmentInfo
    features    :: Vector{<:Base.Callable}
    winparams   :: WinFunction
    treatment   :: Symbol
    modalreduce :: Base.Callable
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

    # run the windowing algo and set windows indexes
    intervals = win(length(X[1,1]))
    n_intervals = length(intervals)

    # define column names and prepare data structure based on treatment type
    if treat == :aggregate        # propositional
        if n_intervals == 1
            # Apply feature to whole time series
            _X = DataFrame(
                [Symbol(f, "(", v, ")") => [f(ts) for ts in X[!, v]]
                    for f in features
                    for v in vnames]...)
            # _X = DataFrame(pairs...)  # Add the splat operator!
        else
            # apply feature to specific intervals
            _X = DataFrame(
                Symbol(f, "(", v, ")w", i) => f.(getindex.(X[!, v], Ref(interval)))
                for f in features
                for v in vnames
                for (i, interval) in enumerate(intervals)
            )
        end

    elseif treat == :reducesize   # modal
        _X = DataFrame(
            [v => [[modalreduce(ts[interval]) for interval in intervals]
                for ts in X[!, v]]
            for v in vnames]...
        )

    else
        error("Unknown treatment type: $treat")
    end

    return _X, TreatmentInfo(features, win, treat, modalreduce)
end