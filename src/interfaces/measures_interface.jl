# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# get_labels(m::AbstractModel) = m.info.supporting_labels
# get_predictions(m::AbstractModel) = m.info.supporting_predictions

# get_labels(ds::AbstractDataset) = string.(ds.ytest)

# # ---------------------------------------------------------------------------- #
# #                                   accuracy                                   #
# # ---------------------------------------------------------------------------- #
# get_accuracy(m::Modelset) = m.results.accuracy

# function get_accuracy(::TypeTreeForestC, m::AbstractModel)
#     MLJ.accuracy(get_labels(m), get_predictions(m))
# end

# function get_accuracy(::TypeModalForest, m::AbstractModel)
#     MLJ.accuracy(get_labels(m.models[1]), get_predictions(m))
# end

# function get_accuracy(::TypeTreeForestC, model::Vector{AbstractModel})
#     MLJ.mean([MLJ.accuracy(get_labels(m), get_predictions(m))
#         for m in model]
#     )
# end

# function get_accuracy(::TypeModalForest, model::Vector{AbstractModel})
#     MLJ.mean([MLJ.accuracy(get_labels(m.models[1]), get_predictions(m))
#         for m in model]
#     )
# end

# # ---------------------------------------------------------------------------- #
# #                              Results structs                                 #
# # ---------------------------------------------------------------------------- #
# struct ClassResults <: AbstractResults
#     accuracy   :: AbstractFloat
#     # rules      :: Union{Rule,          Nothing}
#     # metrics    :: Union{AbstractVector, Nothing}
#     # feature_importance :: Union{AbstractVector, Nothing}
#     # predictions:: Union{AbstractVector, Nothing}
# end

# function ClassResults(
#     setup::AbstractModelSetup{T},
#     model::Union{AbstractModel, Vector{AbstractModel}};
# ) where {T<:AbstractModelType}
#     if model isa Vector{AbstractModel}
#         model = model[1]  # Use the first model in the vector
#     else
#         labels, predictions = get_resultsparams(setup)(model)
#     end
#     labels, predictions = get_resultsparams(setup)(model)
#     accuracy = get_accuracy(model, get_resultsparams(setup))
#     return ClassResults(accuracy)
# end

# # function Base.show(io::IO, ::MIME"text/plain", r::ClassResults)
# #     println(io, "Results")
# #     println(io, "  Accuracy: ", r.accuracy)
# #     println(io, "  Rules: ", r.rules)
# #     println(io, "  Feature importance: ", r.feature_importance)
# #     println(io, "  Predictions: ", r.predictions)
# # end

# # Base.show(io::IO, r::ClassResults) = print(io, "Results(accuracy=$(r.accuracy), rules=$(r.rules))")

# struct RegResults <: AbstractResults
#     accuracy   :: AbstractFloat
#     # rules      :: Union{Rule,          Nothing}
#     # metrics    :: Union{AbstractVector, Nothing}
#     # feature_importance :: Union{AbstractVector, Nothing}
#     # predictions:: Union{AbstractVector, Nothing}
# end

# const RESULTS = Dict{Symbol,DataType}(
#     :classification => ClassResults,
#     :regression     => RegResults
# )

struct Measures <: AbstractMeasures
    per_fold        :: AbstractVector
    measures        :: AbstractVector{<:MLJBase.StatisticalMeasuresBase.Wrapper}
    measures_values :: AbstractVector
    operations      :: AbstractVector{<:Base.Callable}
end