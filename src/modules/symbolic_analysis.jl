# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
const TypeTreeForestC = Union{TypeDTC, TypeRFC, TypeABC}
const TypeTreeForestR = Union{TypeDTR, TypeRFR}

get_algo(setup::AbstractModelSetup) = setup.config.algo

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
    mean([
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
    mean_label = mean(labels)
    total_sum_squares = sum((labels .- mean_label).^2)
    residual_sum_squares = sum((predictions .- labels).^2)
    
    # Handle edge case where variance is close to zero
    # Return R-squared (clamped to prevent extremely negative values)
    return total_sum_squares < 1e-10 ? 0.0 : max(-1.0, 1.0 - (residual_sum_squares / total_sum_squares))
end

function get_accuracy(::TypeTreeForestR, model::Vector{AbstractModel})
    mean([
        begin
            labels = get_labels(m)
            predictions = get_predictions(m)
            
            mean_label = mean(labels)
            total_sum_squares = sum((labels .- mean_label).^2)
            residual_sum_squares = sum((predictions .- labels).^2)
            
            total_sum_squares < 1e-10 ? 0.0 : max(-1.0, 1.0 - (residual_sum_squares / total_sum_squares))
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

# ---------------------------------------------------------------------------- #
#                              symbolic_analysis                               #
# ---------------------------------------------------------------------------- #
function symbolic_analysis(
    X::AbstractDataFrame,
    y::AbstractVector;
    model::Union{NamedTuple, Nothing}=nothing,
    resample::Union{NamedTuple, Nothing}=nothing,
    win::Union{NamedTuple, Nothing}=nothing,
    features::Union{Tuple, Nothing}=nothing,
    tuning::Union{NamedTuple, Bool, Nothing}=nothing,
    rules::Union{NamedTuple, Nothing}=nothing,
    preprocess::Union{NamedTuple, Nothing}=nothing,
    reducefunc::Union{Base.Callable, Nothing}=nothing,
)::Modelset
    # if model is unspecified, use default model setup
    isnothing(model) && (model = DEFAULT_MODEL_SETUP)
    modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, rules, preprocess, reducefunc)
    model = Modelset(modelset, _prepare_dataset(X, y, modelset))
    _traintest!(model)

    # TODO extract rules, if needed

    # save results into model
    model.results = compute_results(model.setup, model.model)

    return model
end

# y is not a vector, but a symbol or a string that identifies a column in X
function symbolic_analysis(
    X::AbstractDataFrame,
    y::Union{Symbol,AbstractString};
    kwargs...
)::Modelset
    symbolic_analysis(X[!, Not(y)], X[!, y]; kwargs...)
end

