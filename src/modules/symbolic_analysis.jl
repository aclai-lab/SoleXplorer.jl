# ---------------------------------------------------------------------------- #
#                              rules extraction                                #
# ---------------------------------------------------------------------------- #
function rules_extraction!(model::Modelset, ds::Dataset, mach::MLJ.Machine)
    model.rules = EXTRACT_RULES[model.setup.rulesparams.type](model, ds, mach)
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
#                                  measures                                    #
# ---------------------------------------------------------------------------- #
function eval_measures!(model::Modelset, y_test::AbstractVector)::Measures
    measures        = MLJBase._actual_measures([get_setup_meas(model)...], get_solemodel(model))
    operations      = get_operations(measures, MLJBase.prediction_type(model.type))

    nfolds          = length(y_test)
    test_fold_sizes = [length(y_test[k]) for k in 1:nfolds]
    nmeasures       = length(get_setup_meas(model))

    # weights used to aggregate per-fold measurements, which depends on a measures
    # external mode of aggregation:
    fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
    fold_weights(::MLJBase.StatisticalMeasuresBase.Sum) = nothing
    
    measurements_vector = mapreduce(vcat, 1:nfolds) do k
        yhat_given_operation = Dict(op=>op(model.model[k], y_test[k]) for op in unique(operations))

        test = y_test[k]

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
        m = get_setup_meas(model)[k]
        mode = MLJBase.StatisticalMeasuresBase.external_aggregation_mode(m)
        MLJBase.StatisticalMeasuresBase.aggregate(
            fold[k];
            mode,
            weights=fold_weights(mode)
        )
    end

    model.measures = Measures(fold, measures, measures_values, operations)
end

# ---------------------------------------------------------------------------- #
#                               get predictions                                #
# ---------------------------------------------------------------------------- #
get_y_test(model::AbstractModel)  = model.info.supporting_labels
get_y_test_folds(model::Modelset) = [get_y_test(m) for m in model.model]

function sole_predict(solem::AbstractModel, y_test)
    classes_seen = unique(y_test)
    eltype(solem.info.supporting_predictions) <: SoleModels.CLabel ?
        begin
            preds = categorical(solem.info.supporting_predictions, levels=levels(classes_seen))
            [UnivariateFinite([p], [1.0]) for p in preds]
        end :
        solem.info.supporting_predictions
end
sole_predict_mode(solem::AbstractModel, y_test) = solem.info.supporting_predictions

# ---------------------------------------------------------------------------- #
#                              symbolic_analysis                               #
# ---------------------------------------------------------------------------- #
function symbolic_analysis(args...; extract_rules::NamedTupleBool=false, kwargs...)
    model, ds = _prepare_dataset(args...; extract_rules, kwargs...)
    mach = _train_machine!(model, ds)
    _test_model!(model, mach, ds)

    if !isa(extract_rules, Bool) || extract_rules
        rules_extraction!(model, ds, mach)
    end

    get_measures(model.setup) === nothing || begin
        y_test = if haskey(model.model[1].info, :supporting_labels)
            get_y_test_folds(model)
        else
            @views [String.(ds.y[i.test]) for i in ds.tt]
        end
        eval_measures!(model, y_test)
    end

    return model, mach, ds
end
