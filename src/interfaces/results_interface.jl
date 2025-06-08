# ---------------------------------------------------------------------------- #
#                              Results structs                                 #
# ---------------------------------------------------------------------------- #
struct ClassResults <: AbstractResults
    accuracy   :: AbstractFloat
    # rules      :: Union{Rule,          Nothing}
    # metrics    :: Union{AbstractVector, Nothing}
    # feature_importance :: Union{AbstractVector, Nothing}
    # predictions:: Union{AbstractVector, Nothing}
end

# function Base.show(io::IO, ::MIME"text/plain", r::ClassResults)
#     println(io, "Results")
#     println(io, "  Accuracy: ", r.accuracy)
#     println(io, "  Rules: ", r.rules)
#     println(io, "  Feature importance: ", r.feature_importance)
#     println(io, "  Predictions: ", r.predictions)
# end

# Base.show(io::IO, r::ClassResults) = print(io, "Results(accuracy=$(r.accuracy), rules=$(r.rules))")

struct RegResults <: AbstractResults
    accuracy   :: AbstractFloat
    # rules      :: Union{Rule,          Nothing}
    # metrics    :: Union{AbstractVector, Nothing}
    # feature_importance :: Union{AbstractVector, Nothing}
    # predictions:: Union{AbstractVector, Nothing}
end

const RESULTS = Dict{Symbol,DataType}(
    :classification => ClassResults,
    :regression     => RegResults
)

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
get_labels(model::AbstractModel) = model.info.supporting_labels
get_predictions(model::AbstractModel) = model.info.supporting_predictions

# ---------------------------------------------------------------------------- #
#                                   accuracy                                   #
# ---------------------------------------------------------------------------- #
get_accuracy(model::Modelset) = model.results.accuracy

function get_accuracy(::TypeTreeForestC, model::AbstractModel)
    labels = get_labels(model)
    predictions = get_predictions(model)
    sum(predictions .== labels)/length(labels)
end

function get_accuracy(::TypeTreeForestC, model::Vector{AbstractModel})
    MLJ.mean([
        begin 
            labels = get_labels(m)
            predictions = get_predictions(m)
            sum(predictions .== labels)/length(labels) 
        end 
        for m in model]
    )
end

function get_accuracy(::TypeTreeForestR, model::AbstractModel)
    labels = get_labels(model)
    predictions = get_predictions(model)
    
    # Calculate R-squared: 1 - (sum of squared residuals / total sum of squares)
    mean_label = MLJ.mean(labels)
    total_sum_squares = sum((labels .- mean_label).^2)
    residual_sum_squares = sum((predictions .- labels).^2)
    
    return total_sum_squares < 1e-10 ? 0.0 : max(-1.0, 1.0 - (residual_sum_squares / total_sum_squares))
end

function get_accuracy(::TypeTreeForestR, model::Vector{AbstractModel})
    MLJ.mean([
        begin
            labels = get_labels(m)
            predictions = get_predictions(m)
            
            mean_label = MLJ.mean(labels)
            total_sum_squares = sum((labels .- mean_label).^2)
            residual_sum_squares = sum((predictions .- labels).^2)
            
            total_sum_squares < 1e-10 ? 0.0 : max(-1.0, 1.0 - (residual_sum_squares / total_sum_squares))
        end
        for m in model]
    )
end

function get_accuracy(::TypeModalForest, model::AbstractModel)
    labels = get_labels(model.models[1])
    predictions = get_predictions(model)
    sum(predictions .== labels)/length(labels)
end

function get_accuracy(::TypeModalForest, model::Vector{AbstractModel})
    MLJ.mean([
        begin
            labels = get_labels(m.models[1])
            predictions = get_predictions(m)
            sum(predictions .== labels)/length(labels) 
        end 
        for m in model]
    )
end

# ---------------------------------------------------------------------------- #
#                               compute_results                                #
# ---------------------------------------------------------------------------- #
function compute_results(
    setup::AbstractModelSetup{T},
    model::Union{AbstractModel, Vector{AbstractModel}};
) where {T<:AbstractModelType}
    accuracy = get_accuracy(T(), model)

    return RESULTS[get_algo(setup)](accuracy)
end