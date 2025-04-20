using SoleXplorer
using DataFrames

X, y = SoleXplorer.@load_iris
X = SoleXplorer.DataFrame(X)

const RULES = Dict(
    :intrees => m -> begin
        if isnothing(m.setup.resample)
            df = DataFrame(m.ds.Xtest, m.ds.info.vnames)
            rules = SoleXplorer.intrees(m.model, df, m.ds.ytest; silent=true, rng=Xoshiro(11))
        else
            rules = reduce(vcat, map(enumerate(m.model)) do (i, model)
                df = DataFrame(m.ds.Xtest[i], m.ds.info.vnames)
                SoleXplorer.intrees(model, df, m.ds.ytest[i]; silent=true, rng=Xoshiro(11))
            end)
        end
    end
)

solem = symbolic_analysis(X, y; model=(type=:decisiontree, params=(;max_depth=6)), preprocess=(;rng=Xoshiro(11)))
rules = RULES[:intrees](solem)

solem = symbolic_analysis(X, y; model=(type=:randomforest, params=(;max_depth=6)), resample=(type=CV,), preprocess=(;rng=Xoshiro(11)))
rules = RULES[:intrees](solem)

solem = symbolic_analysis(X, y; model=(type=:xgboost, params=(max_depth=6, objective="multi:softprob",)),
    tuning=(method=(;type=latinhypercube), params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:eta, lower=0.1, upper=0.5),
            SoleXplorer.range(:num_round, lower=10, upper=80),
        )
    ), preprocess=(;rng=Xoshiro(11)))
rules = RULES[:intrees](solem)