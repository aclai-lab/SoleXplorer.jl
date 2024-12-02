function get_predict(
    mach::Union{MLJ.Machine, AbstractVector{MLJ.Machine}}, 
    valid_X::DataFrame,
    y::CategoricalArray,
    tt_pairs::Union{TTIdx, AbstractVector{TTIdx}}
)
    mach isa MLJ.Machine && (mach = [mach])
    valid_tt = tt_pairs isa TTIdx ? [tt_pairs] : tt_pairs

    test_model = []
    for (i, tt) in enumerate(valid_tt)
        preds = MLJ.predict(mach[i], selectrows(valid_X, tt.test))
        yhat = MLJ.mode.(preds)
        a = MLJ.accuracy(yhat, categorical(y[tt.test]))
        push!(test_model, a)
    end

    return length(test_model) == 1 ? test_model[1] : test_model
end