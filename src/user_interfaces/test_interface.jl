function get_test(
    X::DataFrame,
    y::CategoricalArray,
    tt_pairs::AbstractVector{TTIdx},
    model::T,
    mach
) where {T<:SoleXplorer.ModelConfig}
    result = DecisionTree[]
    
    for (i, tt) in enumerate(tt_pairs)
        learned_dt_tree = haskey(fitted_params(mach[i]), :best_fitted_params) ? fitted_params(mach[i]).best_fitted_params : fitted_params(mach[i])

        if model.classifier isa ModalDecisionTrees.MLJInterface.ModalDecisionTree
            _, sole_dt = report(mach[i]).sprinkle(X[tt.test, :], y[tt.test])
        else
            sole_dt = solemodel(learned_dt_tree.tree)
            apply!(sole_dt, X[tt.test, :], y[tt.test])
        end

        push!(result, sole_dt)
    end

    return result
end