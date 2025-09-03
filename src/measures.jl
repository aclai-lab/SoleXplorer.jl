# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractMeasures

Base type for performance measure containers.
"""
abstract type AbstractMeasures end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const RobustMeasure = StatisticalMeasures.StatisticalMeasuresBase.RobustMeasure
const FussyMeasure  = StatisticalMeasures.StatisticalMeasuresBase.FussyMeasure

const EitherMeasure = Union{RobustMeasure, FussyMeasure}

const ValidMeasures = Union{
        Float64, 
        StatisticalMeasures.ConfusionMatrices.ConfusionMatrix
    }

# ---------------------------------------------------------------------------- #
#                                  measures                                    #
# ---------------------------------------------------------------------------- #
"""
    Measures <: AbstractMeasures

Container for performance evaluation results across CV folds.

# Fields
- `per_fold::Vector{Vector{ValidMeasures}}`: Measure values for each fold/measure combination
- `measures::Vector{RobustMeasure}`: The measure functions used for evaluation  
- `measures_values::Vector{ValidMeasures}`: Aggregated measure values across folds
- `operations::AbstractVector`: Prediction operations used (predict, predict_mode, etc.)
"""
struct Measures <: AbstractMeasures
    per_fold        :: Vector{Vector{ValidMeasures}}
    measures        :: Vector{RobustMeasure}
    measures_values :: Vector{ValidMeasures}
    operations      :: AbstractVector
end

function Base.show(io::IO, m::Measures)
    print(io, "Measures(")
    for (i, (measure, value)) in enumerate(zip(m.measures, m.measures_values))
        if i > 1
            print(io, ", ")
        end
        print(io, "$(measure) = $(value)")
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::Measures)
    println(io, "Measures:")
    for (measure, value) in zip(m.measures, m.measures_values)
        println(io, "  $(measure) = $(value)")
    end
end

# ---------------------------------------------------------------------------- #
#                                  default                                     #
# ---------------------------------------------------------------------------- #
"""
    _DefaultMeasures(y::AbstractVector)::Tuple{Vararg{FussyMeasure}}

Return default measures appropriate for the target variable type.

This function is used when no explicit measures are provided,
automatically selecting between classification and regression.
"""
function _DefaultMeasures(y::AbstractVector)::Tuple{Vararg{FussyMeasure}}
    return eltype(y) <: CLabel ? (accuracy, kappa) : (rms, l1, l2)
end

# ---------------------------------------------------------------------------- #
#                               get operations                                 #
# ---------------------------------------------------------------------------- #
"""
    get_operations(measures::Vector, prediction::Symbol) -> Vector{Function}

Adapted from MLJ's evaluate
Determine appropriate prediction operations for each measure.
"""
function get_operations(
    measures   :: Vector,
    prediction :: Symbol,
)
    map(measures) do m
        kind_of_proxy = MLJBase.StatisticalMeasuresBase.kind_of_proxy(m)
        observation_scitype = MLJBase.StatisticalMeasuresBase.observation_scitype(m)
        isnothing(kind_of_proxy) && (return sole_predict)

        if prediction === :probabilistic
            if kind_of_proxy === MLJBase.LearnAPI.Distribution()
                return sole_predict
            elseif kind_of_proxy === MLJBase.LearnAPI.Point()
                if observation_scitype <: Union{Missing,Finite}
                    return sole_predict_mode
                elseif observation_scitype <:Union{Missing,Infinite}
                    return sole_predict_mean
                else
                    throw(err_ambiguous_operation(prediction, m))
                end
            else
                throw(err_ambiguous_operation(prediction, m))
            end
        elseif prediction === :deterministic
            if kind_of_proxy === MLJBase.LearnAPI.Distribution()
                throw(err_incompatible_prediction_types(prediction, m))
            elseif kind_of_proxy === MLJBase.LearnAPI.Point()
                return sole_predict
            else
                throw(err_ambiguous_operation(prediction, m))
            end
        elseif prediction === :interval
            if kind_of_proxy === MLJBase.LearnAPI.ConfidenceInterval()
                return sole_predict
            else
                throw(err_ambiguous_operation(prediction, m))
            end
        else
            throw(MLJBase.ERR_UNSUPPORTED_PREDICTION_TYPE)
        end
    end
end
