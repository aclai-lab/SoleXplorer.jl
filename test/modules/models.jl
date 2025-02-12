using Test
using Sole, SoleXplorer
using DataFrames
using MLJ, MLJTuning
using Random
using Statistics

X, y = Sole.load_arff_dataset("NATOPS")
train_seed = 11
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# ---------------------------------------------------------------------------- #
models = (
    type=:decisiontree_classifier,
    params=(max_depth=3, min_samples_leaf=14),
    winparams=(type=movingwindow, window_size=12),
    features=[minimum, mean, cov, mode_5]
)

global_params = (
    params=(min_samples_split=17,),
    winparams=(type=adaptivewindow,),
    features=[std]
)

modelsets = validate_modelset(models, typeof(y), global_params)

@test modelsets isa Vector{<:SoleXplorer.AbstractModelSet}
@test modelsets isa Vector{SoleXplorer.SymbolicModelSet}

for m in modelsets
    _classifier = getmodel(m)
    @test _classifier isa MLJ.Model
end

classifier = getmodel(modelsets[1])

@test classifier.max_depth == 3
@test classifier.min_samples_leaf == 14
@test classifier.min_samples_split == 17

# ---------------------------------------------------------------------------- #
models = [(
        type=:decisiontree_classifier,
        params=(max_depth=3, min_samples_leaf=14),
        winparams=(type=movingwindow, window_size=12),
        features=[minimum, mean, cov, mode_5],
        tuning=(
            method=(type=latinhypercube, ntour=20,), 
            params=(repeats=11,), 
            ranges=[SoleXplorer.range(:feature_importance; values=[:impurity, :split])]
        ),   
    ),
    (type=:decisiontree_classifier, params=(min_samples_leaf=30, min_samples_split=3,)
)]

modelsets = validate_modelset(models, typeof(y))

@test modelsets isa Vector{<:SoleXplorer.AbstractModelSet}
@test modelsets isa Vector{SoleXplorer.SymbolicModelSet}

for m in modelsets
    _classifier = getmodel(m)
    @test _classifier isa MLJ.Model
end

classifier = getmodel(modelsets[1])

@test classifier isa MLJTuning.ProbabilisticTunedModel
@test classifier.model isa MLJ.Model
@test classifier.model.max_depth == 3
@test classifier.model.min_samples_leaf == 14
@test classifier.tuning isa LatinHypercube
@test classifier.tuning.ntour == 20
@test classifier.repeats == 11
@test classifier.range[1].field == :feature_importance
@test classifier.range[1].values == (:impurity, :split)


