# ---------------------------------------------------------------------------- #
#                              rules extraction                                #
# ---------------------------------------------------------------------------- #
function rules_extraction!(model::Modelset, ds::Dataset)
    model.rules = EXTRACT_RULES[model.setup.rulesparams.type](model, ds)
end

# ---------------------------------------------------------------------------- #
#                                  measures                                    #
# ---------------------------------------------------------------------------- #
function eval_measures!(model::Modelset)::Measures
    _measures = MLJBase._actual_measures([get_setup_meas(model)...], get_solemodel(model))
    _operations = MLJBase._actual_operations(nothing, _measures, get_mach_model(model), 0)

    y = get_mach_y(model)
    tt = get_setup_tt(model)
    nfolds = length(tt)
    test_fold_sizes = [length(tt[k][1]) for k in 1:nfolds]

    nmeasures = length(get_setup_meas(model))

    # weights used to aggregate per-fold measurements, which depends on a measures
    # external mode of aggregation:
    fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
    fold_weights(::MLJBase.StatisticalMeasuresBase.Sum) = nothing


    measurements_vector = mapreduce(vcat, 1:nfolds) do k
        yhat_given_operation = Dict(op=>op(get_mach(model), rows=tt[k][1]) for op in unique(_operations))
        test = tt[k][1]

        [map(_measures, _operations) do m, op
            m(
                yhat_given_operation[op],
                y[test],
                # MLJBase._view(weights, test),
                # class_weights
                MLJBase._view(nothing, test),
                nothing
            )
        end]
    end

    measurements_matrix = permutedims(reduce(hcat, measurements_vector))

    # measurements for each fold:
    _fold = map(1:nmeasures) do k
        measurements_matrix[:,k]
    end

    # overall aggregates:
    _measures_values = map(1:nmeasures) do k
        m = get_setup_meas(model)[k]
        mode = MLJBase.StatisticalMeasuresBase.external_aggregation_mode(m)
        MLJBase.StatisticalMeasuresBase.aggregate(
            _fold[k];
            mode,
            weights=fold_weights(mode)
        )
    end

    model.measures = Measures(_fold, _measures, _measures_values, _operations)
end

# ---------------------------------------------------------------------------- #
#                              symbolic_analysis                               #
# ---------------------------------------------------------------------------- #
function symbolic_analysis(args...; extract_rules::NamedTupleBool=false, kwargs...)
    model, ds = _prepare_dataset(args...; extract_rules, kwargs...)
    _traintest!(model, ds)

    if !isa(extract_rules, Bool) || extract_rules
        rules_extraction!(model, ds)
    end

    eval_measures!(model)

    return model
end
# function symbolic_analysis(
#     X             :: AbstractDataFrame,
#     y             :: AbstractVector;
#     model         :: NamedTuple     = (;type=:decisiontree),
#     resample      :: NamedTuple     = (;type=Holdout),
#     win           :: OptNamedTuple  = nothing,
#     features      :: OptTuple       = nothing,
#     tuning        :: NamedTupleBool = false,
#     extract_rules :: NamedTupleBool = false,
#     preprocess    :: OptNamedTuple  = nothing,
#     reducefunc    :: OptCallable    = nothing
# )::Modelset
#     modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, extract_rules, preprocess, reducefunc)
#     model = Modelset(modelset, _prepare_dataset(X, y, modelset))
#     _traintest!(model)

#     if !isa(extract_rules, Bool) || extract_rules
#         rules_extraction!(model)
#     end

#     # save results into model
#     # model.results = RESULTS[get_algo(model.setup)](model.setup, model.model)

#     return model
# end

# # y is not a vector, but a symbol or a string that identifies a column in X
# function symbolic_analysis(
#     X::AbstractDataFrame,
#     y::SymbolString;
#     kwargs...
# )::Modelset
#     symbolic_analysis(X[!, Not(y)], X[!, y]; kwargs...)
# end

