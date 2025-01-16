function get_predict(
    mach::MLJ.Machine,
    ds::Dataset,
    kwargs...
)

    preds = MLJ.predict(mach, selectrows(ds.X, ds.tt.test))
    yhat = MLJ.mode.(preds)
    MLJ.accuracy(yhat, categorical(ds.y[ds.tt.test]))
end