```@meta
CurrentModule = SoleXplorer
```

# [Serialization](@id serialize)

SoleXplorer provides serialization functionality using the [JLD2](https://github.com/JuliaIO/JLD2.jl) format.
It enables saving and loading of datasets, models, and analysis results with automatic file
naming conventions and path management.

# Supported Types
All serialization functions work with types that implement the `Saveable` union:
- [`AbstractDataSet`](@ref): Dataset configurations and ML pipelines
- [`AbstractSoleModel`](@ref): Trained symbolic models  
- [`AbstractModelSet`](@ref): Complete analysis results with multiple models

# File Format
All files are saved in JLD2 format with automatic `.jld2` extension handling.

# Naming Convention
Files are automatically prefixed based on content type:
- Datasets: `soleds_<name>.jld2`
- Models: `solemodel_<name>.jld2` 
- Analysis: `soleanalysis_<name>.jld2`

```@docs
solesave
soleload
```

# Examples
```julia
using Test
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

path = @__DIR__

# save dataset setup
r1 = SX.range(:(oversampler.k), lower=3, upper=10)
r2 = SX.range(:(undersampler.min_ratios), lower=0.1, upper=0.9)

dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(max_depth=3),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2))
)
solesave(dsc; path, name="test1")

solemc = train_test(dsc)
solesave(solemc; path, name="test1.jld2")

modelc = symbolic_analysis(
    dsc, solemc,
    extractor=LumenRuleExtractor(minimization_scheme=:mitespresso),
    measures=(accuracy, log_loss, kappa)
)
solesave(modelc; path, name="test1")

# load dataset setup
ds_name        = "soleds_test1"
solemodel_name = "solemodel_test1.jld2"
analysis_name  = "soleanalysis_test1"

dsc_loaded      = soleload(path, ds_name)
model_loaded    = soleload(path, solemodel_name)
analysis_loaded = soleload(path, analysis_name)
```

See also: [`solesave`](@ref), [`soleload`](@ref)
"""