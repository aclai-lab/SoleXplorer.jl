```@meta
CurrentModule = SoleXplorer
```

# [Setup dataset](@id setup-dataset)

```@docs
setup_dataset()
setup_dataset(X::AbstractDataFrame, y::Symbol)
```

# [Dataset](@id dataset)
```@docs
AbstractDataSet
PropositionalDataSet
ModalDataSet
DataSet
get_X(ds::AbstractDataSet)
get_X(ds::AbstractDataSet, part::Symbol)
get_y(ds::AbstractDataSet)
get_y(ds::AbstractDataSet, part::Symbol)
get_mach(ds::AbstractDataSet)
get_mach_model(ds::AbstractDataSet)
get_logiset(ds::ModalDataSet)
```

# [Utilities](@id utilities)
```@docs
code_dataset(X::AbstractDataFrame)
code_dataset(y::AbstractVector)
code_dataset(X::AbstractDataFrame, y::AbstractVector)
```
