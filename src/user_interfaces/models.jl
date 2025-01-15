# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
function get_model(
    model::ModelConfig,
)
    model
end

function get_model(
    modelset::AbstractModelSet,
    ds::Dataset;
    kwargs...
)
    classifier = modelset.model(; modelset.params...)

    if !isnothing(modelset.tuning.method)
        ranges = [r(classifier) for r in modelset.tuning.ranges]

        @show ranges

        classifier = MLJ.TunedModel(; 
            model=classifier, 
            tuning=modelset.tuning.method, 
            range=ranges, 
            modelset.tuning.params...
        )
    end

    return classifier
end
