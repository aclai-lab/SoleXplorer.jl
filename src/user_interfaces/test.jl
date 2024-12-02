function get_test(
    model::T,
    X::DataFrame,
    y::CategoricalArray,
    tt_pairs::Union{TTIdx, AbstractVector{TTIdx}},
    mach::Union{MLJ.Machine, AbstractVector{MLJ.Machine}};
) where {T<:SoleXplorer.ModelConfig}
    mach isa MLJ.Machine && (mach = [mach])
    valid_tt = tt_pairs isa TTIdx ? [tt_pairs] : tt_pairs

    result = DecisionTree[]
    
    for (i, tt) in enumerate(valid_tt)
        learned_dt_tree = haskey(fitted_params(mach[i]), :best_fitted_params) ? fitted_params(mach[i]).best_fitted_params : fitted_params(mach[i])

        if model.classifier isa ModalDecisionTrees.MLJInterface.ModalDecisionTree
            _, sole_dt = report(mach[i]).sprinkle(X[tt.test, :], y[tt.test])
        elseif model.classifier isa MLJTuning.ProbabilisticTunedModel && model.classifier.model isa ModalDecisionTrees.MLJInterface.ModalDecisionTree
            _, sole_dt = report(mach[i])[4].sprinkle(X[tt.test, :], y[tt.test])
        else
            sole_dt = solemodel(learned_dt_tree.tree)
            apply!(sole_dt, selectrows(X, tt.test), y[tt.test])
        end

        push!(result, sole_dt)
    end

    return length(result) == 1 ? result[1] : result
end