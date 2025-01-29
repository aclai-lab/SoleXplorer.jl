# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
get_model(model::ModelConfig) = model

function get_model(modelset::AbstractModelSet)
    classifier = modelset.type(; modelset.params...)

    if modelset.tuning.tuning
        ranges = [r(classifier) for r in modelset.tuning.ranges]

        classifier = MLJ.TunedModel(; 
            model=classifier, 
            tuning=modelset.tuning.method, 
            range=ranges, 
            modelset.tuning.params...
        )
    end

    return classifier
end
