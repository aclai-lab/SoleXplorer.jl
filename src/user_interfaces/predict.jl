function get_predict(
    model::ModelConfig,
    kwargs...
)

    preds = MLJ.predict(model.mach, model.ds.Xtest)
    yhat = MLJ.mode.(preds)
    kp = MLJ.kappa(yhat, model.ds.ytest)
    acc = MLJ.accuracy(yhat, model.ds.ytest)
end