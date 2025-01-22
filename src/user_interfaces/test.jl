function modeltest(
    modelset::AbstractModelSet,
    mach::MLJ.Machine,
    ds::Dataset
)
        modelset.learn_method(mach, selectrows(ds.X, ds.tt.test), ds.y[ds.tt.test])
end