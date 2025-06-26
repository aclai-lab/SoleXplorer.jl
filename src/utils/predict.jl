get_y_test(solem::AbstractModel)  = solem.info.supporting_labels
get_y_test_folds(model::Modelset) = [m.info.supporting_labels for m in model.model]

function sole_predict(solem::AbstractModel)
    classes_seen = unique(solem.info.supporting_labels)
    eltype(solem.info.supporting_predictions) <: SoleModels.CLabel ?
        begin
            preds = categorical(solem.info.supporting_predictions, levels=levels(classes_seen))
            [UnivariateFinite([p], [1.0]) for p in preds]
        end :
        solem.info.supporting_predictions
end
sole_predict_mode(solem::AbstractModel) = solem.info.supporting_predictions
