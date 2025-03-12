function get_predict(
    mach::MLJ.Machine,
    ds::Dataset
)
    preds = MLJ.predict(mach, ds.Xtest)
    yhat = MLJ.mode.(preds)
    kp = MLJ.kappa(yhat, ds.ytest)
    acc = MLJ.accuracy(yhat, ds.ytest)
end