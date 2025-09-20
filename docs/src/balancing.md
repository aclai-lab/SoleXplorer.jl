```@meta
CurrentModule = SoleXplorer
```

# [Balancing](@id balancing)

SoleXplorer uses the following balancing strategies taken from package [Imbalance](https://github.com/JuliaAI/Imbalance.jl): 

[BorderlineSMOTE1]()
[ClusterUndersampler]()
[ENNUndersampler]()
[ROSE]()
[RandomOversampler]()
[RandomUndersampler]()
[RandomWalkOversampler]()
[SMOTE]()
[SMOTEN]()
[SMOTENC]()
[TomekUndersampler]()


```
balancing = (;
    oversample=strategy(; kwargs...),
    undersample=strategy(; kwargs...),
) -> Balancing
```

you can also tune balancing strategy parameters using tuning and ranges, i.e:
```
r1 = SX.range(:(oversampler.k), lower=3, upper=10)
r2 = SX.range(:(undersampler.min_ratios), lower=0.1, upper=0.9)
modelc = symbolic_analysis(
    Xc, yc;
    model=RandomForestClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2)),
    measures=(accuracy, )
)
```