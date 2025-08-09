# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractMeasures end
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
const OptRules    = Optional{DecisionSet}
const OptMeasures = Optional{Measures}

# ---------------------------------------------------------------------------- #
#                                  modelset                                    #
# ---------------------------------------------------------------------------- #
mutable struct ModelSet{S} <: AbstractModelSet
    ds       :: EitherDataSet
    sole     :: Vector{AbstractModel}
    rules    :: OptRules
    measures :: OptMeasures

    function ModelSet(
        ds       :: EitherDataSet,
        sole     :: SoleModel{S};
        rules    :: OptRules = nothing,
        measures :: OptMeasures = nothing
    ) where S
        new{S}(ds, solemodels(sole), rules, measures)
    end
end

function Base.show(io::IO, m::ModelSet{S}) where S
    print(io, "ModelSet{$S}(")
    print(io, "models=$(length(m.sole))")
    if !isnothing(m.rules)
        print(io, ", rules=$(length(m.rules.rules))")
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
        println(io, "  Rules: $(length(m.rules.rules)) extracted rules")
    else
        println(io, "  Rules: none")
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
function supporting_predictions(solem::AbstractModel)
    return solem.info isa Base.RefValue ?
        solem.info[].supporting_predictions :
        solem.info.supporting_predictions
end

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

sole_predict_mode(solem::AbstractModel, y_test::AbstractVector{<:Label}) = supporting_predictions(solem)

function _DefaultMeasures(y::AbstractVector)::Tuple{Vararg{FussyMeasure}}
    return eltype(y) <: CLabel ? (accuracy, kappa) : (rms, l1, l2)
end

# ---------------------------------------------------------------------------- #
#                               get operations                                 #
# ---------------------------------------------------------------------------- #
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
function eval_measures(
    ds::EitherDataSet,
    solem::SoleModel,
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
        yhat_given_operation = Dict(op=>op(solemodels(solem)[k], y_test[k]) for op in unique(operations))

        # Forced to convert it to string as certain StatisticalMeasures measures don't accept
        # categorical arrays, such as confusion matrix and kappa
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
function _symbolic_analysis(
    ds::EitherDataSet,
    solem::SoleModel;
    extractor::Union{Nothing,RuleExtractor,Tuple{RuleExtractor,NamedTuple}}=nothing,
    measures::Tuple{Vararg{FussyMeasure}}=(),
# )::ModelSet
)
    rules = isnothing(extractor)  ? nothing : begin
        # TODO propaga rng, dovrai fare intrees mutable struct
        if extractor isa Tuple
            params = last(extractor)
            extractor = first(extractor)
        else
            params = NamedTuple(;)
        end
        extractrules(extractor, params, ds, solem)
    end

    y_test = get_y_test(ds)
    isempty(measures) && (measures = _DefaultMeasures(first(y_test)))
    # all_classes = unique(Iterators.flatten(y_test))
    measures = eval_measures(ds, solem, measures, y_test)

    # return ModelSet(ds, solem; rules, measures)
    return rules
end

function symbolic_analysis(
    ds::EitherDataSet,
    solem::SoleModel;
    kwargs...
# )::ModelSet
)
    _symbolic_analysis(ds, solem; kwargs...)
end

function symbolic_analysis(
    X::AbstractDataFrame,
    y::AbstractVector,
    w::OptVector = nothing;
    extractor::Union{Nothing,RuleExtractor}=nothing,
    measures::Tuple{Vararg{FussyMeasure}}=(),
    kwargs...
)::ModelSet
    ds = _setup_dataset(X, y, w; kwargs...)
    solem = _train_test(ds)
    _symbolic_analysis(ds, solem; extractor, measures)
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
solemodels(m::ModelSet) = m.sole
rules(m::ModelSet) = m.rules
