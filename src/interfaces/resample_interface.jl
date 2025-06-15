# ---------------------------------------------------------------------------- #
#                                     types                                    #
# ---------------------------------------------------------------------------- #
struct Resample <: AbstractTypeParams
    type        :: Base.Callable
    params      :: NamedTuple
end

# ---------------------------------------------------------------------------- #
#                                   resample                                   #
# ---------------------------------------------------------------------------- #
const AVAIL_RESAMPLES = (CV, Holdout, StratifiedCV, TimeSeriesCV)

const RESAMPLE_PARAMS = Dict{DataType,NamedTuple}(
    CV           => (
        nfolds         = 6,
        shuffle        = true,
        rng            = TaskLocalRNG()
    ),
    Holdout      => (
        fraction_train = 0.7,
        shuffle        = true,
        rng            = TaskLocalRNG()
    ),
    StratifiedCV => (
        nfolds         = 6,
        shuffle        = true,
        rng            = TaskLocalRNG()
    ),
    TimeSeriesCV => (
        nfolds         = 4,
    )
)
