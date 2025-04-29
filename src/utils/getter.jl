# ---------------------------------------------------------------------------- #
#                               Modelset setup                                 #
# ---------------------------------------------------------------------------- #


function resample_guard(model::Union{AbstractModel, Vector{AbstractModel}}, label::Symbol)
    model isa DecisionEnsemble && begin
     return [m.info[label] for m in model.models]
    end
    model isa Vector ? [m.info[label] for m in model] : model.info[label]
end



# ---------------------------------------------------------------------------- #
#                              Modelset results                                #
# ---------------------------------------------------------------------------- #

