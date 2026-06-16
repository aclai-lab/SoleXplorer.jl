```@meta
CurrentModule = SoleXplorer
```

# Dataset

This page documents the dataset setup and configuration API.

## Overview

The dataset layer wraps an MLJ machine together with partition indices
and metadata. It is the central data structure passed through the entire
SoleXplorer workflow.

## Entry Point

```@docs
setup_dataset
```

## DataSet

```@docs
DataSet
```

### Accessors

```@docs
get_X
get_y
get_mach
get_mach_model
get_logiset
```

## Partitioning

Partitioning splits data into train/validation/test sets according to
an MLJ resampling strategy.

```@docs
partition
get_train
get_valid
get_test
```

### Parametrized Cross-Validation

```@docs
pCV
```

## ModelSet

```@docs
AbstractModelSet
ModelSet
```
