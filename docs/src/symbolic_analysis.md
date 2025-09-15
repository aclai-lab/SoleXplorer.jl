```@meta
CurrentModule = SoleXplorer
```

# [Symbolic analysis](@id symbolic-analysis)

This is the entry point of SoleXplorer: this function can be used standalone, for finalize an already trained model, or to update already analyzed results.

```@docs
symbolic_analysis(X::AbstractDataFrame, y::AbstractVector, w::MaybeVector)
symbolic_analysis(ds::AbstractDataSet, solem::SoleModel)
symbolic_analysis!(modelset::ModelSet)
```

# [ModelSet](@id ModelSet)

```@docs
AbstractModelSet
ModelSet
dsetup(m::ModelSet)
solemodels(m::ModelSet)
rules(m::ModelSet)
associations(m::ModelSet)
performance(m::ModelSet)
```


