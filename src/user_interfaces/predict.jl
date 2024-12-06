function get_predict(
    model::T, 
    ds::S,
    kwargs...
) where {T<:SoleXplorer.ModelConfig, S<:SoleXplorer.Dataset}
    mach = model.mach isa MLJ.Machine ? [model.mach] : model.mach
    tt_test = ds.tt isa AbstractVector ? ds.tt : [ds.tt]

    test_model = []
    for (i, tt) in enumerate(tt_test)
        preds = MLJ.predict(mach[i], selectrows(ds.X, tt.test))
        yhat = MLJ.mode.(preds)
        a = MLJ.accuracy(yhat, categorical(ds.y[tt.test]))
        push!(test_model, a)
    end

    return length(test_model) == 1 ? test_model[1] : test_model
end