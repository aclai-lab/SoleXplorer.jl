# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractTuning end

const MaybeResamplingStrategy = Maybe{MLJ.ResamplingStrategy}

# ---------------------------------------------------------------------------- #
#                             MLJ Tuning adapter                               #
# ---------------------------------------------------------------------------- #
struct Tuning <: AbstractTuning
    strategy::MLJ.TuningStrategy
    resampling::MaybeResamplingStrategy
    measure::EitherMeasure
    # weights=nothing,
    # class_weights=nothing,
    operations=nothing,
    operation=operations,
    ranges=nothing,
    range=ranges,
    selection_heuristic=NaiveSelection(),
    train_best=true,
    repeats=1,
    n=nothing,
    acceleration=default_resource(),
    acceleration_resampling=CPU1(),
    check_measure=true,
    cache=true,
    compact_history=true,
    logger=MLJBase.default_logger()
end

const MaybeTuning = Maybe(Tuning)