# ---------------------------------------------------------------------------- #
#                               Modelset setup                                 #
# ---------------------------------------------------------------------------- #
get_algo(model::Modelset) = model.setup.config.algo
get_labels(model::Modelset) = model.model.info.supporting_labels
get_predictions(model::Modelset) = model.model.info.supporting_predictions