"""
symbolic_analysis.jl — Unified Symbolic Model Analysis Interface

This module provides the main entry point for complete symbolic model analysis workflows:

1. Dataset setup and cross-validation training (via train_test.jl)
2. Training and testing models (via train_test.jl)
3. Rule extraction from symbolic models (via extractrules.jl) 
4. Performance evaluation using MLJ measures
"""

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractMeasures

Base type for performance measure containers.
"""
abstract type AbstractMeasures end

"""
    AbstractModelSet  

Base type for comprehensive analysis result containers.
"""
abstract type AbstractModelSet end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const RobustMeasure = StatisticalMeasures.StatisticalMeasuresBase.RobustMeasure
const FussyMeasure  = StatisticalMeasures.StatisticalMeasuresBase.FussyMeasure

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
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const MayRules        = Maybe{Union{Vector{DecisionSet}, Vector{LumenResult}}}
const MayMeasures     = Maybe{Measures}
const MayAssociations = Maybe{Vector{ARule}}

# ---------------------------------------------------------------------------- #
#                                  modelset                                    #
# ---------------------------------------------------------------------------- #
"""
    ModelSet{S} <: AbstractModelSet

Comprehensive container for symbolic model analysis results.

# Fields
- `ds::EitherDataSet`: Dataset wrapper used for training
- `sole::Vector{AbstractModel}`: Symbolic models from each CV fold
- `rules::MayRules`: Extracted decision rules (optional)
- `measures::MayMeasures`: Performance evaluation results (optional)
"""
mutable struct ModelSet{S} <: AbstractModelSet
    ds           :: EitherDataSet
    sole         :: Vector{AbstractModel}
    rules        :: MayRules
    associations :: MayAssociations
    measures     :: MayMeasures

    function ModelSet(
        ds       :: EitherDataSet,
        sole     :: SoleModel{S};
        rules    :: MayRules=nothing,
        miner    :: MayAssociations=nothing,
        measures :: MayMeasures=nothing
    ) where S
        new{S}(ds, solemodels(sole), rules, miner, measures)
    end
end

function Base.show(io::IO, m::ModelSet{S}) where S
    print(io, "ModelSet{$S}(")
    print(io, "models=$(length(m.sole))")
    if !isnothing(m.rules)
        print(io, ", rules=$(length(m.rules.rules))")
    end
        if !isnothing(m.associations)
        print(io, ", associations=$(length(m.associations))")
    end
    if !isnothing(m.measures)
        print(io, ", measures=$(length(m.measures.measures))")
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::ModelSet{S}) where S
    println(io, "ModelSet{$S}:")
    println(io, "  Dataset: $(typeof(m.ds))")
    println(io, "  Models: $(length(m.sole)) symbolic models")
    
    if !isnothing(m.rules)
        println(io, "  Rules: $(length(first(m.rules))) extracted rules per model")
    else
        println(io, "  Rules: none")
    end

    if !isnothing(m.associations)
        println(io, "  Associations: $(length(m.associations)) associated rules per model")
    else
        println(io, "  Associations: none")
    end
    
    if !isnothing(m.measures)
        println(io, "  Measures:")
        for (measure, value) in zip(m.measures.measures, m.measures.measures_values)
            println(io, "    $(measure) = $(value)")
        end
    else
        println(io, "  Measures: none")
    end
end

# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
"""
    supporting_predictions(solem::AbstractModel) -> Vector

Extract supporting predictions from a symbolic model.
"""
function supporting_predictions(solem::AbstractModel)
    return solem.info isa Base.RefValue ?
        solem.info[].supporting_predictions :
        solem.info.supporting_predictions
end

"""
    sole_predict(solem::AbstractModel, y_test::AbstractVector{<:Label}) -> Vector

Convert symbolic model predictions to MLJ probabilistic format.
"""
function sole_predict(solem::AbstractModel, y_test::AbstractVector{<:Label})
    preds = supporting_predictions(solem)
    eltype(preds) <: CLabel ?
        begin
            classes_seen = unique(y_test)
            eltype(preds) <: MLJ.CategoricalValue ||
                (preds = categorical(preds, levels=levels(classes_seen)))
            [UnivariateFinite([p], [1.0]) for p in preds]
        end :
        preds
end

"""
    sole_predict_mode(solem::AbstractModel, y_test::AbstractVector{<:Label}) -> Vector

Return deterministic predictions from symbolic model.
"""
sole_predict_mode(solem::AbstractModel, y_test::AbstractVector{<:Label}) = supporting_predictions(solem)

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

# ---------------------------------------------------------------------------- #
#                                eval measures                                 #
# ---------------------------------------------------------------------------- #
"""
    eval_measures(ds::EitherDataSet, solem::Vector{AbstractModel}, 
                 measures::Tuple{Vararg{FussyMeasure}}, 
                 y_test::Vector{<:AbstractVector{<:Label}}) -> Measures

Adapted from MLJ's evaluate
Evaluate symbolic models using MLJ measures across CV folds.
"""
function eval_measures(
    ds::EitherDataSet,
    solem::Vector{AbstractModel},
    measures::Tuple{Vararg{FussyMeasure}},
    y_test::Vector{<:AbstractVector{<:Label}}
)::Measures
    mach_model = get_mach_model(ds)
    measures        = MLJBase._actual_measures([measures...], mach_model)
    operations      = get_operations(measures, MLJBase.prediction_type(mach_model))

    nfolds          = length(ds)
    test_fold_sizes = [length(y_test[k]) for k in 1:nfolds]
    nmeasures       = length(measures)

    # weights used to aggregate per-fold measurements, which depends on a measures
    # external mode of aggregation:
    fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
    fold_weights(::MLJBase.StatisticalMeasuresBase.Sum) = nothing
    
    measurements_vector = mapreduce(vcat, 1:nfolds) do k
        yhat_given_operation = Dict(op=>op(solem[k], y_test[k]) for op in unique(operations))

        # costretto a convertirlo a stringa in quanto certe misure di statistical measures non accettano
        # categorical array, tipo confusion matrix e kappa
        test = eltype(y_test[k]) <: CLabel ? String.(y_test[k]) : y_test[k]

        [map(measures, operations) do m, op
            m(
                yhat_given_operation[op],
                test,
                # MLJBase._view(weights, test),
                # class_weights
                MLJBase._view(nothing, test),
                nothing
            )
        end]
    end

    measurements_matrix = permutedims(reduce(hcat, measurements_vector))

    # measurements for each fold:
    fold = map(1:nmeasures) do k
        measurements_matrix[:,k]
    end

    # overall aggregates:
    measures_values = map(1:nmeasures) do k
        m = measures[k]
        mode = MLJBase.StatisticalMeasuresBase.external_aggregation_mode(m)
        MLJBase.StatisticalMeasuresBase.aggregate(
            fold[k];
            mode,
            weights=fold_weights(mode)
        )
    end

    Measures(fold, measures, measures_values, operations)
end

# ---------------------------------------------------------------------------- #
#                              symbolic_analysis                               #
# ---------------------------------------------------------------------------- #
function _symbolic_analysis!(
    modelset::ModelSet;
    extractor::Union{Nothing,RuleExtractor,Tuple{RuleExtractor,NamedTuple}}=nothing,
    association::Union{Nothing,AbstractAssociationExtractor}=nothing,
    measures::Tuple{Vararg{FussyMeasure}}=()
)::Nothing
    ds = dsetup(modelset)
    solem = solemodels(modelset)

    modelset.rules = isnothing(extractor) ? nothing : begin
        # TODO propaga rng, dovrai fare intrees mutable struct
        if extractor isa Tuple
            params = last(extractor)
            extractor = first(extractor)
        else
            params = NamedTuple(;)
        end
        extractrules(extractor, params, ds, solem)
    end

    modelset.associations = isnothing(association) ? nothing : begin
        # X = scalarlogiset(get_X(ds))
        X = get_logiset(ds)
        algo = get_method(association)
        masargs = get_mas_args(association)
        maskwargs = get_mas_kwargs(association)

        Miner(X |> deepcopy, algo, masargs...; maskwargs...) |> mine!
    end

    y_test = get_y_test(ds)
    isempty(measures) && (measures = _DefaultMeasures(first(y_test)))
    # all_classes = unique(Iterators.flatten(y_test))
    modelset.measures = eval_measures(ds, solem, measures, y_test)

    return nothing
end

function _symbolic_analysis(
    ds::EitherDataSet,
    solem::SoleModel;
    kwargs...
)::ModelSet
    modelset = ModelSet(ds, solem)
    _symbolic_analysis!(modelset; kwargs...)
    return modelset
end

"""
    symbolic_analysis(ds::EitherDataSet, solem::SoleModel; 
                     extractor=nothing, measures=()) -> ModelSet

Perform symbolic analysis on pre-trained models.

Use when you already have trained symbolic models and want to add
rule extraction and/or performance evaluation.
"""
function symbolic_analysis(
    ds::EitherDataSet,
    solem::SoleModel;
    kwargs...
)::ModelSet
    _symbolic_analysis(ds, solem; kwargs...)
end

function symbolic_analysis!(
    modelset::ModelSet; 
    kwargs...
)::ModelSet
    _symbolic_analysis!(modelset; kwargs...)
    return modelset
end

"""
    symbolic_analysis(X::AbstractDataFrame, y::AbstractVector, w=nothing;
                     extractor=nothing, measures=(), kwargs...) -> ModelSet

End-to-end symbolic analysis starting from raw data.

# Arguments
- `X, y, w`: Features, targets, and optional weights
- `extractor`: Rule extraction strategy (ModalExtractor, etc.)
- `measures`: Performance measures to evaluate (accuracy, auc, etc.)
- `kwargs`: Passed to dataset setup (model, cv_folds, etc.)

See [`setup_dataset`](@ref) for dataset setup parameter descriptions.

# Extended help

## Workflow
1. `_setup_dataset(X, y, w; kwargs...)` - Create dataset wrapper
2. `_train_test(ds)` - Perform CV training and symbolic conversion
3. `_symbolic_analysis(ds, solem; extractor, measures)` - Extract rules and evaluate
"""
function symbolic_analysis(
    X::AbstractDataFrame,
    y::AbstractVector,
    w::MayVector = nothing;
    extractor::Union{Nothing,RuleExtractor}=nothing,
    association::Union{Nothing,AbstractAssociationExtractor}=nothing,
    measures::Tuple{Vararg{FussyMeasure}}=(),
    kwargs...
)::ModelSet
    ds = _setup_dataset(X, y, w; kwargs...)
    solem = _train_test(ds)
    _symbolic_analysis(ds, solem; extractor, association, measures)
end

symbolic_analysis(X::Any, args...; kwargs...) = symbolic_analysis(DataFrame(X), args...; kwargs...)

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
"""
    dsetup(m::ModelSet) -> EitherDataSet

Extract the dataset setup from a ModelSet.
"""
dsetup(m::ModelSet) = m.ds

"""
    solemodels(m::ModelSet) -> Vector{AbstractModel}

Extract the vector of symbolic models from a ModelSet.
"""
solemodels(m::ModelSet) = m.sole

"""
    rules(m::ModelSet) -> DecisionSet

Extract the vector of rules from a ModelSet.
"""
rules(m::ModelSet) = m.rules

"""
    associations(m::ModelSet) -> Vector{ARule}

Extract the vector of associations from a ModelSet.
"""
associations(m::ModelSet) = m.associations
