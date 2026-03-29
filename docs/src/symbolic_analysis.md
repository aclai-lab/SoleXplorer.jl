```@meta
CurrentModule = SoleXplorer
```

# [Symbolic analysis](@id symbolic-analysis)

This is the entry point of SoleXplorer: this function can be used standalone, for finalize an already trained model, or to update already analyzed results.

```@docs
solexplorer(X::AbstractDataFrame, y::AbstractVector, w::Union{Nothing,Vector})
solexplorer(ds::AbstractDataSet, solem::SoleModel)
solexplorer!(modelset::ModelSet)
```

# [ModelSet](@id ModelSet)

```@docs
AbstractModelSet
ModelSet
dsetup(m::ModelSet)
solemodels(m::ModelSet)
rules(m::ModelSet)
<!-- associations(m::ModelSet) -->
performance(m::ModelSet)
```


