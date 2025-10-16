# experimental
# the goal is to find a valid interface to implement a timeout algo

struct TimeOut
    training     :: Float64
    rules        :: Float64
    associations :: Float64

    function TimeOut(;
        training     :: Float64=Inf,
        rules        :: Float64=Inf,
        associations :: Float64=Inf,
    )
        new(training, rules, associations)
    end
end

function TimeOut(;
    training     :: Number=Inf,
    rules        :: Number=Inf,
    associations :: Number=Inf,    
)
    TimeOut(;
        training=Float64(training),
        rules=Float64(rules),
        associations=Float64(associations))
end

# example:
# modelr = symbolic_analysis(
#     Xr, yr;
#     model=DecisionTreeRegressor(),
#     resampling=CV(nfolds=5, shuffle=true),
#     seed=1,
#     tuning=GridTuning(; range, resolution=10, resampling=CV(nfolds=3), measure=rms, repeats=2),
#     measures=(rms, l1, l2, mae, mav)
#     extractor=LumenRuleExtractor()
#     timeout=(;rules=15)
# )

# means that timeout is only applied to rule_extraction and should be of 15 minutes.
# you con enter any type of number: it will be converted to Float64 to stay adhere to the algo

# another aspect that must be covered is how algo reacts if timeout is reached.
# ideally, would be awesome if it automatically tune the parameters and starts another run.

# like: random forest take to long to complete with 100 trees, at failing reduce the trees.
# a first thought could be to make a dictionary where at every algo is associated one or more 
# sensible parameter.
# is random forest, the number of trees is timewise sensible.