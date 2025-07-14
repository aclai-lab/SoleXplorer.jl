# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractMeasures end
# abstract type AbstractSolePrediction end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const RobustMeasure = MLJ.StatisticalMeasures.StatisticalMeasuresBase.RobustMeasure
const FussyMeasure = MLJ.StatisticalMeasures.StatisticalMeasuresBase.FussyMeasure

# ---------------------------------------------------------------------------- #
#                                  measures                                    #
# ---------------------------------------------------------------------------- #
mutable struct Measures <: AbstractMeasures
    per_fold        :: Vector{Vector{Float64}}
    measures        :: Vector{RobustMeasure}
    measures_values :: Vector{Float64}
    operations      :: AbstractVector

    # function Measures(
    #     yhat      :: AbstractVector,
    # )::Measures
    #     new(yhat, nothing, nothing, nothing, nothing)
    # end
end


# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #

# # ---------------------------------------------------------------------------- #
# #                              rules extraction                                #
# # ---------------------------------------------------------------------------- #
# function rules_extraction!(model::Modelset, ds::Dataset, mach::MLJ.Machine)
#     model.rules = EXTRACT_RULES[model.setup.rulesparams.type](model, ds, mach)
# end

# ---------------------------------------------------------------------------- #
#                             get sole predictions                             #
# ---------------------------------------------------------------------------- #
supporting_predictions(solem::AbstractModel) = solem.info.supporting_predictions

function sole_predict(solem::AbstractModel, y_test::AbstractVector{<:Label})
    preds = supporting_predictions(solem)
    eltype(preds) <: SoleModels.CLabel ?
        begin
            classes_seen = unique(y_test)
            preds = categorical(preds, levels=levels(classes_seen))
            [UnivariateFinite([p], [1.0]) for p in preds]
        end :
        preds
end

sole_predict_mode(solem::AbstractModel, y_test::AbstractVector{<:Label}) = supporting_predictions(solem)

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
    model::EitherDataSet,
    solem::ModelSet,
    measures::Tuple{Vararg{FussyMeasure}},
    y_test::Vector{<:AbstractVector{<:Label}}
)::Measures
    measures        = MLJBase._actual_measures([measures...], solem)
    operations      = get_operations(measures, MLJBase.prediction_type(get_mach_model(model)))

    nfolds          = length(model)
    test_fold_sizes = [length(y_test[k]) for k in 1:nfolds]
    nmeasures       = length(measures)

    # weights used to aggregate per-fold measurements, which depends on a measures
    # external mode of aggregation:
    fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
    fold_weights(::MLJBase.StatisticalMeasuresBase.Sum) = nothing
    
    measurements_vector = mapreduce(vcat, 1:nfolds) do k
        yhat_given_operation = Dict(op=>op(solem.sole[k], y_test[k]) for op in unique(operations))

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
function _symbolic_analysis(
    model::EitherDataSet,
    solem::ModelSet;
    # extract_rules=true,
    measures::Tuple{Vararg{FussyMeasure}}=(),
)::Measures
    if !isempty(measures)
        y_test = get_y_test(model)
        eval_measures(model, solem, measures, y_test)
    else
        thow(ArgumentError("No measures provided for symbolic analysis."))
    end
end

function symbolic_analysis(
    model::EitherDataSet,
    solem::ModelSet;
    kwargs...
)::Measures
    _symbolic_analysis(model, solem; kwargs...)
end

function symbolic_analysis(
    X::AbstractDataFrame,
    y::AbstractVector;
    measures::Tuple{Vararg{FussyMeasure}}=(),
    kwargs...
)::Measures
    model = _prepare_dataset(X, y; kwargs...)
    solem = _train_test(model)
    _symbolic_analysis(model, solem; measures)
end
