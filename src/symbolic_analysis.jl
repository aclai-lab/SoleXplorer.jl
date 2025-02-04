function _traintest(
    X::AbstractDataFrame, 
    y::Union{AbstractVector, Nothing}; 
    models::AbstractVector{<:NamedTuple}, 
    global_params::Union{NamedTuple, Nothing}=nothing,
    preprocess_params::Union{NamedTuple, Nothing}=nothing,
    kwargs...
)
    modelsets = validate_modelset(models, global_params, preprocess_params)

    models = ModelConfig[]

    for m in modelsets
        ds = prepare_dataset(X, y, m)

        classifier = getmodel(m)

        mach = fitmodel(m, classifier, ds);
        model = testmodel(m, mach, ds);

        push!(models, ModelConfig(m, ds, classifier, mach, model))
    end
    return models
end

function traintest(
    X::AbstractDataFrame, 
    y::AbstractVector; 
    models::Union{NamedTuple, AbstractVector{<:NamedTuple}, Nothing}=nothing, 
    kwargs...
)
    check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))

    isnothing(models) && throw(ArgumentError("At least one type must be specified"))

    if isa(models, NamedTuple)
        first(_traintest(X, y; models=[models], kwargs...))
    else
        _traintest(X, y; models=models, kwargs...)
    end
end

function symbolic_analysis(
        X::AbstractDataFrame, 
        y::AbstractVector; 
        models::Union{NamedTuple, AbstractVector{<:NamedTuple}, Nothing}=nothing, 
        kwargs...
    )

end

# function _symbolic_analysis(
#     X::AbstractDataFrame, 
#     y::Union{AbstractVector, Nothing}; 
#     models::AbstractVector{<:NamedTuple}, 
#     global_params::Union{NamedTuple, Nothing}=nothing,
#     preprocess_params::Union{NamedTuple, Nothing}=nothing,
#     kwargs...
# )
#     modelsets = validate_modelset(models, global_params, preprocess_params)

#     models = ModelConfig[]

#     for m in modelsets
#         ds = prepare_dataset(X, y, m)

#         classifier = getmodel(m)

#         mach = fitmodel(m, classifier, ds);
#         model = testmodel(m, mach, ds);

#         rules = get_rules(m, model, ds);
#         accuracy = get_predict(mach, ds);

#         push!(models, ModelConfig(m, ds, classifier, mach, model, rules, accuracy))
#     end
#     return models
# end

# function symbolic_analysis(
#     X::AbstractDataFrame, 
#     y::AbstractVector; 
#     models::Union{NamedTuple, AbstractVector{<:NamedTuple}, Nothing}=nothing, 
#     kwargs...
# )
#     check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
#     size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))

#     isnothing(models) && throw(ArgumentError("At least one type must be specified"))

#     if isa(models, NamedTuple)
#         _symbolic_analysis(X, y; models=[models], kwargs...)
#     else
#         _symbolic_analysis(X, y; models=models, kwargs...)
#     end
# end # Use a NamedTuple type