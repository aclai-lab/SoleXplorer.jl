function _symbolic_analysis!(tt::AbstractVector{ModelConfig})
    for t in tt
        t.rules = SolePostHoc.modalextractrules(
            t.setup.rulesparams.type,
            t.model,
            t.ds.Xtrain,
            t.ds.ytrain;
            t.setup.rulesparams.params...
        )
        t.accuracy = get_predict(t.mach, t.ds)
    end
end

function symbolic_analysis(
    X::AbstractDataFrame, 
    y::AbstractVector; 
    models::Union{NamedTuple, AbstractVector{<:NamedTuple}, Nothing}=nothing, 
    globals::Union{NamedTuple, Nothing}=nothing,
    preprocess::Union{NamedTuple, Nothing}=nothing,
# )::Union{ModelConfig, AbstractVector{ModelConfig}}
)
    tt = traintest(X, y; models, globals, preprocess)

    if isa(tt, AbstractVector)
        _symbolic_analysis!(tt)
    else
        _symbolic_analysis!([tt])
    end

    return tt
end

