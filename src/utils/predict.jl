get_y_test(solem::AbstractModel)  = solem.info.supporting_labels
get_y_test_folds(model::Modelset) = [m.info.supporting_labels for m in model.model]

# get_y_test(::DecisionTreeApply, solem::AbstractModel)  = solem.info.supporting_labels
# get_y_test_folds(::DecisionTreeApply, model::Modelset) = [m.info.supporting_labels for m in model.model]

function sole_predict(solem::AbstractModel)
    classes_seen = categorical(unique(solem.info.supporting_labels))
    predictions  = categorical(solem.info.supporting_predictions, levels=levels(classes_seen))
    [UnivariateFinite([p], [1.0]) for p in predictions]
end
sole_predict_mode(solem::AbstractModel) = solem.info.supporting_predictions


# function sole_predict(
#     ::MLJ.Machine{<:MLJDecisionTreeInterface.DecisionTreeClassifier,<:Any,true}, 
#     solem::AbstractModel
# )
#     classes_seen = categorical(unique(solem.info.supporting_labels))
#     predictions  = categorical(solem.info.supporting_predictions, levels=levels(classes_seen))
#     [UnivariateFinite([p], [1.0]) for p in predictions]
# end
# sole_predict_mode(::DecisionTreeApply, solem::AbstractModel) = solem.info.supporting_predictions
