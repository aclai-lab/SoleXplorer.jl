function get_predict(
    mach::Union{MLJ.Machine, AbstractVector{MLJ.Machine}}, 
    valid_X, y, tt_pairs
)
    mach isa MLJ.Machine && (mach = [mach])

    test_model = []
    for (i, tt) in enumerate(tt_pairs)
        preds = MLJ.predict(mach[i], valid_X[tt.test, :])
        yhat = MLJ.mode.(preds)
        a = MLJ.accuracy(yhat, categorical(y[tt.test]))
        push!(test_model, a)
    end

    return length(test_model) == 1 ? test_model[1] : test_model
end