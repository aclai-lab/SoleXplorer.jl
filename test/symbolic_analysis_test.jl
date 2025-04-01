using SoleXplorer
using DataFrames
using DecisionTree: load_data

X, y = load_data("iris")
X = DataFrame(Float64.(X), :auto)
X=X[:, 1:3]
y = String.(y)
rng = Xoshiro(11)

using SolePostHoc

# modelset = train_test(X, y; model=(type=:decisiontree,))
modelset = train_test(X, y; model=(type=:randomforest, params=(n_trees=3,)))

extractor = SolePostHoc.LumenRuleExtractor()

### working ###
rules1 = SolePostHoc.modalextractrules(
    extractor,
    modelset.mach.fitresult[1],
    apply_function=nothing
)

### failing ###
rules2 = SolePostHoc.modalextractrules(
    extractor,
    modelset.model,
)

### working ###
# test = lumen(
#     modelset.mach.fitresult[1]
# )

### failing ###
test = lumen(
    modelset.model
)