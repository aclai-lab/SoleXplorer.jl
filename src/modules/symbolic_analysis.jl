# ---------------------------------------------------------------------------- #
#                              rules extraction                                #
# ---------------------------------------------------------------------------- #
function rules_extraction!(model::Modelset)
    model.rules = EXTRACT_RULES[model.setup.rulesparams.type](model)
end

# ---------------------------------------------------------------------------- #
#                                  measures                                    #
# ---------------------------------------------------------------------------- #
# function eval_measures!(model::Modelset)
#     # mach::Machine,
#     # resampling,
#     # weights,
#     # class_weights,
#     # rows,
#     # verbosity,
#     # repeats,
#     # measures,
#     # operations,
#     # acceleration,
#     # force,
#     # per_observation_flag,
#     # logger,
#     # user_resampling,
#     # compact,
#     # )

#     # Note: `user_resampling` keyword argument is the user-defined resampling strategy,
#     # while `resampling` is always a `TrainTestPairs`.

#     # Note: `rows` and `repeats` are only passed to the final `PeformanceEvaluation`
#     # object to be returned and are not otherwise used here.

#     # if !(resampling isa TrainTestPairs)
#     #     error("`resampling` must be an "*
#     #           "`MLJ.ResamplingStrategy` or tuple of rows "*
#     #           "of the form `(train_rows, test_rows)`")
#     # end

#     # X = mach.args[1]()
#     # y = mach.args[2]()
#     # nrows = MLJBase.nrows(y)

#     # nfolds = length(resampling)
#     # test_fold_sizes = map(resampling) do train_test_pair
#     #     test = last(train_test_pair)
#     #     test isa Colon && (return nrows)
#     #     length(test)
#     # end

#     # weights used to aggregate per-fold measurements, which depends on a measures
#     # external mode of aggregation:
#     # fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
#     # fold_weights(::StatisticalMeasuresBase.Sum) = nothing

#     # nmeasures = length(measures)

#     # function fit_and_extract_on_fold(mach, k)
#     #     train, test = resampling[k]
#     #     fit!(mach; rows=train, verbosity=verbosity - 1, force=force)
#     #     # build a dictionary of predictions keyed on the operations
#     #     # that appear (`predict`, `predict_mode`, etc):
#     #     yhat_given_operation =
#     #         Dict(op=>op(mach, rows=test) for op in unique(operations))

#     #     ytest = selectrows(y, test)
#     #     if per_observation_flag
#     #         measurements =  map(measures, operations) do m, op
#     #             StatisticalMeasuresBase.measurements(
#     #                 m,
#     #                 yhat_given_operation[op],
#     #                 ytest,
#     #                 _view(weights, test),
#     #                 class_weights,
#     #             )
#     #         end
#     #     else
#     #         measurements =  map(measures, operations) do m, op
#     #             m(
#     #                 yhat_given_operation[op],
#     #                 ytest,
#     #                 _view(weights, test),
#     #                 class_weights,
#     #             )
#     #         end
#     #     end

#     #     fp = fitted_params(mach)
#     #     r = report(mach)
#     #     return (measurements, fp, r)
#     # end

#     # if acceleration isa CPUProcesses
#     #     if verbosity > 0
#     #         @info "Distributing evaluations " *
#     #               "among $(nworkers()) workers."
#     #     end
#     # end
#     #  if acceleration isa CPUThreads
#     #     if verbosity > 0
#     #         nthreads = Threads.nthreads()
#     #         @info "Performing evaluations " *
#     #           "using $(nthreads) thread" * ifelse(nthreads == 1, ".", "s.")
#     #     end
#     # end

#     # measurements_vector_of_vectors, fitted_params_per_fold, report_per_fold  =
#     #     _evaluate!(
#     #         fit_and_extract_on_fold,
#     #         mach,
#     #         acceleration,
#     #         nfolds,
#     #         verbosity
#     #     )

#     measurements_vector_of_vectors = mapreduce(vcat, 1:nfolds) do k
#         yhat_given_operation =
#             Dict(op=>op(mach, rows=test) for op in unique(operations))

#         ytest = selectrows(y, test)
#         if per_observation_flag
#             measurements =  map(measures, operations) do m, op
#                 StatisticalMeasuresBase.measurements(
#                     m,
#                     yhat_given_operation[op],
#                     ytest,
#                     _view(weights, test),
#                     class_weights,
#                 )
#             end
#         else
#             measurements =  map(measures, operations) do m, op
#                 m(
#                     yhat_given_operation[op],
#                     ytest,
#                     _view(weights, test),
#                     class_weights,
#                 )
#             end
#         end
#         return measurements
#     end

#     # return zip(ret...) |> collect

#     measurements_flat = vcat(measurements_vector_of_vectors...)

#     # In the `measurements_matrix` below, rows=folds, columns=measures; each element of
#     # the matrix is:
#     #
#     # - a vector of meausurements, one per observation within a fold, if
#     # - `per_observation_flag = true`; or
#     #
#     # - a single measurment for the whole fold, if `per_observation_flag = false`.
#     #
#     measurements_matrix = permutedims(
#         reshape(collect(measurements_flat), (nmeasures, nfolds))
#     )

#     # measurements for each observation:
#     per_observation = if per_observation_flag
#        map(1:nmeasures) do k
#            measurements_matrix[:,k]
#        end
#     else
#         fill(missing, nmeasures)
#     end

#     # measurements for each fold:
#     per_fold = if per_observation_flag
#         map(1:nmeasures) do k
#             m = measures[k]
#             mode = StatisticalMeasuresBase.external_aggregation_mode(m)
#             map(per_observation[k]) do v
#                 StatisticalMeasuresBase.aggregate(v; mode)
#             end
#         end
#     else
#         map(1:nmeasures) do k
#             measurements_matrix[:,k]
#         end
#     end

#     # overall aggregates:
#     per_measure = map(1:nmeasures) do k
#         m = measures[k]
#         mode = StatisticalMeasuresBase.external_aggregation_mode(m)
#         StatisticalMeasuresBase.aggregate(
#             per_fold[k];
#             mode,
#             weights=fold_weights(mode),
#         )
#     end

#     evaluation = PerformanceEvaluation(
#         mach.model,
#         measures,
#         per_measure,
#         operations,
#         per_fold,
#         per_observation,
#         fitted_params_per_fold |> collect,
#         report_per_fold |> collect,
#         resampling,
#         user_resampling,
#         repeats
#     )
#     log_evaluation(logger, evaluation)

#     compact && return compactify(evaluation)
#     return evaluation
# end

# ---------------------------------------------------------------------------- #
#                              symbolic_analysis                               #
# ---------------------------------------------------------------------------- #
function symbolic_analysis(args...; extract_rules::NamedTupleBool=false, kwargs...)
    model, ds = _prepare_dataset(args...; kwargs...)
    _traintest!(model, ds)

    if !isa(extract_rules, Bool) || extract_rules
        rules_extraction!(model)
    end

    # save results into model
    # model.results = RESULTS[get_algo(model.setup)](model.setup, model.model)

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

