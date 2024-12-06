function modeltest(
    model::T,
    ds::S,
    kwargs...
) where {T<:SoleXplorer.ModelConfig, S<:SoleXplorer.Dataset}
    mach = model.mach isa MLJ.Machine ? [model.mach] : model.mach
    tt_test = ds.tt isa AbstractVector ? ds.tt : [ds.tt]

    result = DecisionTree[]
    
    for (i, tt) in enumerate(tt_test)
        learned_dt_tree = haskey(MLJ.fitted_params(mach[i]), :best_fitted_params) ? MLJ.fitted_params(mach[i]).best_fitted_params : MLJ.fitted_params(mach[i])

        if model.classifier isa ModalDecisionTrees.MLJInterface.ModalDecisionTree
            _, sole_dt = report(mach[i]).sprinkle(ds.X[tt.test, :], ds.y[tt.test])
        elseif model.classifier isa MLJTuning.ProbabilisticTunedModel && model.classifier.model isa ModalDecisionTrees.MLJInterface.ModalDecisionTree
            _, sole_dt = report(mach[i])[4].sprinkle(ds.X[tt.test, :], ds.y[tt.test])
        else
            sole_dt = solemodel(learned_dt_tree.tree)
            apply!(sole_dt, selectrows(ds.X, tt.test), ds.y[tt.test])
        end

        push!(result, sole_dt)
    end

    return length(result) == 1 ? result[1] : result
end