# ---------------------------------------------------------------------------- #
#                                     types                                    #
# ---------------------------------------------------------------------------- #
"""
    TuningStrategy <: AbstractTypeParams

A structure representing a hyperparameter tuning strategy for machine learning models.

`TuningStrategy` encapsulates a tuning method (such as grid search, random search, 
or particle swarm optimization) along with its specific configuration parameters.

# Fields
- `type::Base.Callable`: The tuning method to use. Should be one of the available methods
  defined in `AVAIL_TUNING_METHODS` such as `grid`, `randomsearch`, `particleswarm`, etc.
- `params::NamedTuple`: Configuration parameters for the specified tuning method.
  Default parameters for each method are available in `TUNING_METHODS_PARAMS`.
"""
struct TuningStrategy <: AbstractTypeParams
    type        :: Base.Callable
    params      :: NamedTuple
end

"""
    TuningParams <: AbstractTypeParams

A structure that fully defines a hyperparameter tuning configuration for machine learning models.

`TuningParams` combines a tuning strategy with tuning parameters and hyperparameter ranges,
providing all necessary information to execute a hyperparameter tuning process.

# Fields
- `method::TuningStrategy`: The tuning strategy to use, which defines both the tuning method
  and its configuration parameters.
- `params::NamedTuple`: Additional parameters for the tuning process, such as the number of
  iterations, resampling strategy, performance measure, etc. Default parameters are available
  in `TUNING_PARAMS` for both classification and regression tasks.
- `ranges::Tuple{Vararg{Base.Callable}}`: A tuple of parameter range functions created using
  the `range` function, which define the hyperparameters to tune and their search spaces.
"""
struct TuningParams <: AbstractTypeParams
    method      :: TuningStrategy
    params      :: NamedTuple
    ranges      :: Tuple{Vararg{Base.Callable}}
end

# ---------------------------------------------------------------------------- #
#                                   tuning                                     #
# ---------------------------------------------------------------------------- #
const AVAIL_TUNING_METHODS  = (grid, randomsearch, latinhypercube, particleswarm, adaptiveparticleswarm)

const TUNING_METHODS_PARAMS = Dict{Union{DataType, UnionAll},NamedTuple}(
    grid                  => (
        goal                   = nothing,
        resolution             = 10,
        shuffle                = true,
        rng                    = TaskLocalRNG()
    ),
    randomsearch          => (
        bounded                = MLJ.Distributions.Uniform,
        positive_unbounded     = MLJ.Distributions.Gamma,
        other                  = MLJ.Distributions.Normal,
        rng                    = TaskLocalRNG()
    ),
    latinhypercube        => (
        gens                   = 1,
        popsize                = 100,
        ntour                  = 2,
        ptour                  = 0.8,
        interSampleWeight      = 1.0,
        ae_power               = 2,
        periodic_ae            = false,
        rng                    = TaskLocalRNG()
    ),
    particleswarm         => (
        n_particles            = 3,
        w                      = 1.0,
        c1                     = 2.0,
        c2                     = 2.0,
        prob_shift             = 0.25,
        rng                    = TaskLocalRNG()
    ),
    adaptiveparticleswarm => (
        n_particles            = 3,
        c1                     = 2.0,
        c2                     = 2.0,
        prob_shift             = 0.25,
        rng                    = TaskLocalRNG()
    )
)

const TUNING_PARAMS = Dict{DataType,NamedTuple}(
    AbstractClassification => (;
        resampling              = Holdout(),
        measure                 = LogLoss(tol = 2.22045e-16),
        weights                 = nothing,
        class_weights           = nothing,
        repeats                 = 1,
        operation               = nothing,
        selection_heuristic     = MLJ.MLJTuning.NaiveSelection(nothing),
        n                       = 25,
        train_best              = true,
        acceleration            = default_resource(),
        acceleration_resampling = CPU1(),
        check_measure           = true,
        cache                   = true,
    ),
    AbstractRegression => (;
        resampling              = Holdout(),
        measure                 = MLJ.RootMeanSquaredError(),
        weights                 = nothing,
        class_weights           = nothing,
        repeats                 = 1,
        operation               = nothing,
        selection_heuristic     = MLJ.MLJTuning.NaiveSelection(nothing),
        n                       = 25,
        train_best              = true,
        acceleration            = default_resource(),
        acceleration_resampling = CPU1(),
        check_measure           = true,
        cache                   = true,
    ),
)

"""
    range(field::Union{Expr, Symbol}; kwargs...) -> Function

Create a hyperparameter range specification function for model tuning.

This function returns a function that, when called with a model, creates an MLJ parameter range
for the specified `field`. This two-step approach allows defining parameter ranges before 
having the actual model instance.

# Arguments
- `field::Union{Expr, Symbol}`: The parameter name to tune, either as a Symbol (e.g., `:max_depth`) 
  or as an expression for nested parameters (e.g., `:(tree.max_depth)`).

# Keyword Arguments
- `lower::Union{AbstractFloat, Int, Nothing}=nothing`: Lower bound for numerical parameters.
- `upper::Union{AbstractFloat, Int, Nothing}=nothing`: Upper bound for numerical parameters.
- `origin::Union{AbstractFloat, Int, Nothing}=nothing`: Reference point for the range.
- `unit::Union{AbstractFloat, Int, Nothing}=nothing`: Unit of measurement for parameters with scale.
- `scale::OptSymbol=nothing`: Scale type (e.g., `:linear`, `:log`, `:logit`).
- `values::Union{AbstractVector, Nothing}=nothing`: Explicit set of values for the parameter.

# Returns
- `Function`: A function that takes a model and returns an MLJ.ParamRange object.

# Examples
```julia
# Create a range for max_depth from 2 to 10
depth_range = range(:max_depth, lower=2, upper=10)

# Create a range for min_samples_leaf with specific values
leaf_range = range(:min_samples_leaf, values=[1, 5, 10, 20])

# Create a log-scale range for learning rate
lr_range = range(:learning_rate, lower=1e-4, upper=0.1, scale=:log)

# Apply to a model
model = DecisionTreeClassifier()
actual_range = depth_range(model)
"""
function range(
    field  :: Union{Expr, Symbol};
    lower  :: Union{AbstractFloat, Int, Nothing} = nothing,
    upper  :: Union{AbstractFloat, Int, Nothing} = nothing,
    origin :: Union{AbstractFloat, Int, Nothing} = nothing,
    unit   :: Union{AbstractFloat, Int, Nothing} = nothing,
    scale  :: OptSymbol             = nothing,
    values :: Union{AbstractVector, Nothing}     = nothing,
)
    return function(model)
        MLJ.range(
            model,
            field;
            lower  = lower,
            upper  = upper,
            origin = origin,
            unit   = unit,
            scale  = scale,
            values = values,
        )
    end
end
