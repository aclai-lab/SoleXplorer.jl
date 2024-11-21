using Sole
using SoleXplorer
using Random, StatsBase

X, y = SoleData.load_arff_dataset("NATOPS");
rng = Random.Xoshiro(1)
features = [minimum, mean, StatsBase.cov, mode_5]
nwindows = 10
overlap = 2

@info "Test: Decision Tree based on world filtering"

# code included in SoleXplorer.jl/src/user_interfaces/worlds_interface.jl
maxsize = argmax(length.(Array(X[!, :])))
intervals = collect(allworlds(frame(X, maxsize.I[1])))

valid_intervals = absolute_movingwindow(intervals, nwindows, overlap)
