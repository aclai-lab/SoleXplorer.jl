```@meta
CurrentModule = SoleXplorer
```

# [Tuning](@id tuning)

SoleXplorer uses the following tuning strategies adapted from package [MLJ](https://juliaai.github.io/MLJ.jl/stable/): `GridTuning`, `RandomTuning`, `CubeTuning`, `ParticleTuning` and `AdaptiveTuning`.

```
strategy_type::Type{<:Any}(;
    range::RangeSpec,
    MLJ.ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true),
    measure::MaybeMeasure=nothing,
    repeats::Int64=1,
    strategy_kwargs...
) -> Tuning
```

# Arguments
- `strategy_type`: Type of optimization strategy to instantiate
- `range`: Parameter ranges to explore
- `resampling`: Cross-validation for hyperparameter evaluation
- `measure`: Performance metric for optimization  
- `repeats`: Number of optimization runs
- `kwargs...`: Strategy-specific parameters

# [Tuning Strategies](@id tuning-strategies)
```@docs
GridTuning
RandomTuning
CubeTuning
ParticleTuning
AdaptiveTuning
```

# [Tuning Range](@id range)

```@docs
range
```