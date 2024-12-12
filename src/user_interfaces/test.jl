function modeltest!(
    model::T,
    ds::S,
    kwargs...
) where {T<:SoleXplorer.ModelConfig, S<:SoleXplorer.Dataset}
    mach = model.mach isa MLJ.Machine ? [model.mach] : model.mach
    tt_test = ds.tt isa AbstractVector ? ds.tt : [ds.tt]

    # result = model.model_type[]
    
    for (i, tt) in enumerate(tt_test)
        learn_apply_method = model.apply_tuning ? model.tune_learn_method : model.learn_method
        sole_dt = learn_apply_method(mach[i], selectrows(ds.X, tt.test), ds.y[tt.test])

        push!(model.rules, sole_dt)
    end

    # return length(result) == 1 ? result[1] : result
end