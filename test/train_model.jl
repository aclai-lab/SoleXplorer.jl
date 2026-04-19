using CSV
using DataFrames
using CategoricalArrays
using SoleXplorer
const SX = SoleXplorer

using SoleModels
const SM = SoleModels

using JSON
using Logging

function json_error(msg)
    println(JSON.json(Dict("success" => false, "error" => msg)))
    exit(1)
end

function parse_positive_int(params, key, default_value)
    if !haskey(params, key)
        return default_value
    end

    parsed = tryparse(Int, params[key])
    if parsed === nothing || parsed < 1
        return default_value
    end

    return parsed
end

function var_index_int64(v)::Int64
    s = v isa Symbol ? String(v) : string(v)
    m = match(r"\d+$", s)
    m === nothing && error("Cannot extract numeric suffix from variable: $v")
    return parse(Int64, m.match)
end

function get_values(r, features)
    if r isa Tuple{Vararg{SyntaxBranch}}
        for s in r
            get_values(s.children, features)
        end
    elseif r isa SyntaxBranch
        get_values(r.children, features)
    elseif r isa Tuple{Vararg{Atom}}
        for a in r
            features[var_index_int64(a.value.metacond.feature.i_variable)] += 1
        end
    elseif r isa Tuple
        for s in r
            get_values(s, features)
        end
    end
end

m_model = ARGS[1]
filename = "/home/paso/Documents/Laravel/wekavel/storage/app/public/" * ARGS[2]
tree_depth, n_trees, early_stopping_rate = parse.(Int, ARGS[3:end])

try
    df = CSV.read(filename, DataFrame)
    X  = select(df, 1:ncol(df)-1)
    y  = CategoricalArrays.categorical(string.(df[!, end]))

    models = Dict(
        "decision_tree" => (X, y) -> SX.solexplorer(
            X, y;
            model=SX.DecisionTreeClassifier(max_depth=tree_depth),
            resampling=Holdout(fraction_train=0.7, shuffle=true),
            extractor=LumenRuleExtractor(),
            seed=42
        ),
        "random_forest" => (X, y) -> SX.solexplorer(
            X, y;
            model=SX.RandomForestClassifier(max_depth=4, n_trees=n_trees),
            resampling=Holdout(fraction_train=0.7, shuffle=true),
            extractor=LumenRuleExtractor(),
            seed=42
        ),
        "ada_boost"     => (X, y) -> SX.solexplorer(
            X, y;
            model=SX.AdaBoostStumpClassifier(n_iter=n_trees),
            resampling=Holdout(fraction_train=0.7, shuffle=true),
            extractor=LumenRuleExtractor(),
            seed=42
        ),
        "xgboost"       => (X, y) -> SX.solexplorer(
            X, y;
            model=SX.XGBoostClassifier(early_stopping_rounds=early_stopping_rate),
            resampling=Holdout(fraction_train=0.7, shuffle=true),
            extractor=LumenRuleExtractor(),
            seed=42
        )
    )

    result   = models[m_model](X, y)

    @show result

    accuracy = SX.values(result)[1]
    kappa    = SX.values(result)[2]

    fnames   = names(X)
    features = zeros(Int64, size(X, 2))
    set = SX.rules(result)[1].decision_set.rules

    for r in set
        if r isa Rule
            get_values(r.antecedent, features)
        end
    end

    # Output as JSON for Laravel
    output = Dict(
        "success"  => true,
        "accuracy" => accuracy,
        "kappa"    => kappa,
        "features" => Vector(features),
        "fnames"   => Vector(fnames)
    )

    println(JSON.json(output))
catch e
    bt = sprint(io -> Base.showerror(io, e, catch_backtrace()))
    println(JSON.json(Dict(
        "success" => false,
        "error" => string(e),
        "backtrace" => bt
    )))
    exit(1)
end
