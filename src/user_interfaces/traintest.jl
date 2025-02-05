function _traintest(
    X::AbstractDataFrame, 
    y::AbstractVector; 
    models::AbstractVector{<:NamedTuple}, 
    globals::Union{NamedTuple, Nothing}=nothing,
    preprocess::Union{NamedTuple, Nothing}=nothing,
)
    modelsets = validate_modelset(models, globals, preprocess)

    models = map(m -> begin
        ds = prepare_dataset(X, y, m)
        classifier = getmodel(m)
        mach = fitmodel(m, classifier, ds)
        model = testmodel(m, mach, ds)
        ModelConfig(m, ds, classifier, mach, model)
    end, modelsets)

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