using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames, Random

using Distributions

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SX.NatopsLoader()
Xts, yts = SX.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                                 grid tuning                                  #
# ---------------------------------------------------------------------------- #
seed = 42
model = SX.DecisionTreeClassifier()
measures=(SX.accuracy, SX.kappa)
resampling = SX.CV(nfolds=10, shuffle=true)
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)

tuning = GridTuning(;
    resolution=7,
    resampling,
    range,
    measure=SX.accuracy
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    seed,
    tuning,
    measures
)

@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.ProbabilisticTunedModel}
@test m.ds.mach.model.tuning isa MLJ.Grid
@test m.ds.mach.model.tuning.goal === nothing
@test m.ds.mach.model.tuning.resolution == 7

# ---------------------------------------------------------------------------- #
seed = 42
model = SX.RandomForestRegressor()
measures=(SX.rms,)
resampling = SX.CV(nfolds=6, shuffle=true)
range1 = SX.range(:n_subfeatures, lower=1, upper=9)
range2 = SX.range(:sampling_fraction, lower=0.4, upper=1.0)

tuning = GridTuning(;
    goal=30,
    repeats=4,
    resampling,
    range=(range1, range2),
    measure=SX.rms
)
m = solexplorer(
    Xr, yr;
    model,
    resampling,
    seed,
    tuning,
    measures
)

@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.DeterministicTunedModel}
@test m.ds.mach.model.tuning isa MLJ.Grid
@test m.ds.mach.model.tuning.goal == 30

# ---------------------------------------------------------------------------- #
tuning = GridTuning(;
    goal=2,
    resampling,
    range=(range1, range2),
    measure=SX.rms
)
@test_throws ArgumentError solexplorer(
    Xr, yr;
    model,
    resampling,
    seed,
    tuning,
    measures
)

seed = 42
tuning = GridTuning(;
    goal=2,
    resampling,
    range=range1,
    measure=SX.rms
)
m = solexplorer(
    Xr, yr;
    model,
    resampling,
    seed,
    tuning,
    measures
)
@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.DeterministicTunedModel}
@test m.ds.mach.model.tuning isa MLJ.Grid
@test m.ds.mach.model.tuning.goal == 2

seed = 42
model = SX.DecisionTreeClassifier()
measures=(SX.accuracy, SX.kappa)
range1 = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
tuning = GridTuning(;
    goal=30,
    resampling,
    range=range1,
    measure=SX.accuracy
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    seed,
    tuning,
    measures
)
@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.ProbabilisticTunedModel}
@test m.ds.mach.model.tuning isa MLJ.Grid
@test m.ds.mach.model.tuning.goal == 30

seed = 42
model = SX.XGBoostClassifier()
resampling = SX.CV(nfolds=3, shuffle=true)
measures = (SX.accuracy, SX.kappa)
range1 = SX.range(:num_round; lower=20, upper=100, unit=20)
range2 = SX.range(:eta; lower=0.2, upper=0.6, unit=0.1)
range3 = SX.range(:max_depth; lower=4, upper=6)

tuning = GridTuning(;
    resolution=4,
    resampling,
    range=(range1,range2,range3),
    measure=SX.accuracy
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    seed,
    tuning,
    measures
)
@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.ProbabilisticTunedModel}
@test m.ds.mach.model.tuning isa MLJ.Grid
@test m.ds.mach.model.tuning.goal === nothing
@test m.ds.mach.model.tuning.resolution == 4

tuning = GridTuning(;
    goal=10,
    resampling,
    range=(range1,range2,range3),
    measure=SX.accuracy
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    seed,
    tuning,
    measures
)
@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.ProbabilisticTunedModel}
@test m.ds.mach.model.tuning isa MLJ.Grid
@test m.ds.mach.model.tuning.goal == 10

# ---------------------------------------------------------------------------- #
#                            random search tuning                              #
# ---------------------------------------------------------------------------- #
seed = 42
model = SX.DecisionTreeClassifier()
measures=(SX.accuracy, SX.kappa)
resampling = SX.CV(nfolds=10, shuffle=true)
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)

tuning = RandomTuning(;
    range,
    measure=SX.accuracy,
    rng=Random.Xoshiro(seed)
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    seed,
    tuning,
    measures
)

@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.ProbabilisticTunedModel}
@test m.ds.mach.model.tuning isa MLJ.RandomSearch
@test m.ds.mach.model.tuning.bounded == Distributions.Uniform
@test m.ds.mach.model.tuning.other == Distributions.Normal
@test m.ds.mach.model.tuning.positive_unbounded == Distributions.Gamma

# ---------------------------------------------------------------------------- #
seed = 42
model = SX.RandomForestRegressor()
measures=(SX.rms,)
resampling = SX.CV(nfolds=6, shuffle=true)
range1 = SX.range(:n_subfeatures, lower=1, upper=9)
range2 = SX.range(:sampling_fraction, lower=0.4, upper=1.0)

tuning = RandomTuning(;
    resampling,
    range=(range1, range2),
    bounded=Distributions.Normal,
    other=Distributions.Gamma,
    positive_unbounded=Distributions.Uniform,
    rng=Random.Xoshiro(seed),
    measure=SX.rms
)
m = solexplorer(
    Xr, yr;
    model,
    resampling,
    seed,
    tuning,
    measures
)

@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.DeterministicTunedModel}
@test m.ds.mach.model.tuning isa MLJ.RandomSearch
@test m.ds.mach.model.tuning.bounded == Distributions.Normal
@test m.ds.mach.model.tuning.other == Distributions.Gamma
@test m.ds.mach.model.tuning.positive_unbounded == Distributions.Uniform

# ---------------------------------------------------------------------------- #
#                            particle swarm tuning                             #
# ---------------------------------------------------------------------------- #
seed = 42
model = SX.DecisionTreeRegressor()
measures=(SX.rms,)
resampling = SX.CV(nfolds=6, shuffle=true)
range1 = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
range2 = SX.range(:post_prune, values=[true, false])
range3 = SX.range(:feature_importance, values=[:impurity, :split])

tuning = ParticleTuning(;
    repeats=2,
    resampling,
    range=range1,
    measure=SX.rms
)
m = solexplorer(
    Xr, yr;
    model,
    resampling,
    seed,
    tuning,
    measures
)

@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.DeterministicTunedModel}
@test m.ds.mach.model.tuning isa SX.MLJParticleSwarmOptimization.ParticleSwarm
@test m.ds.mach.model.tuning.c1 == 2.0
@test m.ds.mach.model.tuning.c2 == 2.0
@test m.ds.mach.model.tuning.n_particles == 3
@test m.ds.mach.model.tuning.prob_shift == 0.25
@test m.ds.mach.model.tuning.w == 1.0

tuning = ParticleTuning(;
    c1=2.2,
    c2=1.8,
    n_particles=4,
    prob_shift=0.35,
    w=1.2,
    rng=Random.Xoshiro(seed),
    repeats=2,
    resampling,
    range=(range1, range2, range3),
    measure=SX.rms
)
m = solexplorer(
    Xr, yr;
    model,
    resampling,
    seed,
    tuning,
    measures
)

@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.DeterministicTunedModel}
@test m.ds.mach.model.tuning isa SX.MLJParticleSwarmOptimization.ParticleSwarm
@test m.ds.mach.model.tuning.c1 == 2.2
@test m.ds.mach.model.tuning.c2 == 1.8
@test m.ds.mach.model.tuning.n_particles == 4
@test m.ds.mach.model.tuning.prob_shift == 0.35
@test m.ds.mach.model.tuning.w == 1.2

# ---------------------------------------------------------------------------- #
#                      adaptive particle swarm tuning                          #
# ---------------------------------------------------------------------------- #
seed = 42
model = SX.DecisionTreeClassifier()
measures=(SX.accuracy, SX.kappa)
resampling = SX.CV(nfolds=10, shuffle=true)
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)

tuning = AdaptiveTuning(;
    repeats=2,
    resampling,
    range=range,
    measure=SX.accuracy
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    seed,
    tuning,
    measures
)

@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.ProbabilisticTunedModel}
@test m.ds.mach.model.tuning isa
    SX.MLJParticleSwarmOptimization.AdaptiveParticleSwarm
@test m.ds.mach.model.tuning.c1 == 2.0
@test m.ds.mach.model.tuning.c2 == 2.0
@test m.ds.mach.model.tuning.n_particles == 3
@test m.ds.mach.model.tuning.prob_shift == 0.25

seed = 42
model = SX.XGBoostClassifier()
resampling = SX.CV(nfolds=3, shuffle=true)
measures = (SX.accuracy, SX.kappa)
range1 = SX.range(:num_round; lower=20, upper=100, unit=20)
range2 = SX.range(:eta; lower=0.2, upper=0.6, unit=0.1)
range3 = SX.range(:max_depth; lower=4, upper=6)

tuning = AdaptiveTuning(;
    c1=2.2,
    c2=1.8,
    n_particles=4,
    prob_shift=0.35,
    rng=Random.Xoshiro(seed),
    repeats=2,
    resampling,
    range=(range1, range2, range3),
    measure=SX.accuracy
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    seed,
    tuning,
    measures
)

@test m isa SX.ModelSet
@test m.ds.mach isa MLJ.Machine{<:MLJ.MLJTuning.ProbabilisticTunedModel}
@test m.ds.mach.model.tuning isa
    SX.MLJParticleSwarmOptimization.AdaptiveParticleSwarm
@test m.ds.mach.model.tuning.c1 == 2.2
@test m.ds.mach.model.tuning.c2 == 1.8
@test m.ds.mach.model.tuning.n_particles == 4
@test m.ds.mach.model.tuning.prob_shift == 0.35
