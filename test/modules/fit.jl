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

@testset "fitmodel Tests" begin

# ---------------------------------------------------------------------------- #
    models = (type=:decisiontree,)
    global_params = (winparams=(type=wholewindow,), features=[mean])

    modelsets = validate_modelset(models, global_params)
    ds = prepare_dataset(X, y, first(modelsets))
    @test ds isa SoleXplorer.Dataset
    classifier = getmodel(first(modelsets))
    @test classifier isa MLJ.Model

    fmodel = fitmodel(first(modelsets), classifier, ds)

    @test fmodel isa MLJ.Machine
    @test fmodel.model isa MLJ.Model

# ---------------------------------------------------------------------------- #
    models = [(
            type=:decisiontree,
            features=[minimum, mean, cov, mode_5],
            tuning=(
                method=(type=latinhypercube, ntour=20,), 
                params=(repeats=11,), 
                ranges=[SoleXplorer.range(:feature_importance; values=[:impurity, :split])]
            ),   
        ),
        (type=:decisiontree, params=(min_samples_leaf=30, min_samples_split=3,)
    )]

    modelsets = validate_modelset(models)
    ds = prepare_dataset(X, y, first(modelsets))
    @test ds isa SoleXplorer.Dataset
    classifier = getmodel(first(modelsets))
    @test classifier isa MLJ.Model

    fmodel = fitmodel(first(modelsets), classifier, ds)

    @test fmodel isa MLJ.Machine
    @test fmodel.model isa MLJ.Model

# ---------------------------------------------------------------------------- #
    models = (type=:decisiontree,)
    global_params = (winparams=(type=wholewindow,), features=[mean])
    preprocess_params = (stratified=true, nfolds=5)

    modelsets = validate_modelset(models, global_params, preprocess_params)
    ds = prepare_dataset(X, y, first(modelsets))
    @test ds isa SoleXplorer.Dataset
    classifier = getmodel(first(modelsets))
    @test classifier isa MLJ.Model

    fmodel = fitmodel(first(modelsets), classifier, ds)

    @test fmodel isa Vector{<:MLJ.Machine}
    @test fmodel[1].model isa MLJ.Model

# ---------------------------------------------------------------------------- #
end



