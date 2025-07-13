# in mlj il tuning avviene definendo dei range di lavoro e creando un modello apposito,
# che contiene tutti i parametri di tuning (resample va visto come una sorta di train/validation),
# da non confondere con il resample originale train/test.
# il problema è che, essendo tutto esterno, cieè: dapprima si crea il modello, poi si applicano i range sul modello,
# e infine si crea il modello di tuning, passando come parametro il modello originale,
# per mantenere pulita l'interfaccia di Sole, occorre separare i parametri del modello e del tuning,
# poi, internamente, se sono presenti parametri di tuning, ricreare il modello di tuning.
# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractTune end

# ---------------------------------------------------------------------------- #
#                                     tune                                     #
# ---------------------------------------------------------------------------- #
struct Tune <: AbstractTune
    type     :: MLJ.MLJTuning.TuningStrategy
    resample :: MLJ.ResamplingStrategy
    ranges   :: Tuple{Vararg{Base.Callable}}
    measure  :: Union{MLJ.Measure, Tuple{Vararg{MLJ.Measure}}}

    function Tune(;
        type     :: MLJ.MLJTuning.TuningStrategy,
        resample :: MLJ.ResamplingStrategy,
        ranges   :: Tuple{Vararg{Base.Callable}},
        measure  :: Union{MLJ.Measure, Tuple{Vararg{MLJ.Measure}}}
    )
        new(type, resample, ranges, measure)
    end
end


function tune(
    model::MLJ.Model,
    tuning::MLJ.MLJTuning.TuningStrategy
)
    return MLJ.TunedModel(; 
            model, 
            tuning,
            range=ranges, 
            model.tuning.params...
        )
end

# # ---------------------------------------------------------------------------- #
# #                                     types                                    #
# # ---------------------------------------------------------------------------- #
# struct TuningStrategy <: AbstractTypeParams
#     type        :: Base.Callable
#     params      :: NamedTuple
# end

# struct TuningParams <: AbstractTypeParams
#     method      :: TuningStrategy
#     params      :: NamedTuple
#     ranges      :: Tuple{Vararg{Base.Callable}}
# end

# # ---------------------------------------------------------------------------- #
# #                                   tuning                                     #
# # ---------------------------------------------------------------------------- #
# const AVAIL_TUNING_METHODS  = (grid, randomsearch, latinhypercube, particleswarm, adaptiveparticleswarm)

# const TUNING_METHODS_PARAMS = Dict{Union{DataType, UnionAll},NamedTuple}(
#     grid                  => (
#         goal                   = nothing,
#         resolution             = 10,
#         shuffle                = true,
#         rng                    = TaskLocalRNG()
#     ),
#     randomsearch          => (
#         bounded                = MLJ.Distributions.Uniform,
#         positive_unbounded     = MLJ.Distributions.Gamma,
#         other                  = MLJ.Distributions.Normal,
#         rng                    = TaskLocalRNG()
#     ),
#     latinhypercube        => (
#         gens                   = 1,
#         popsize                = 100,
#         ntour                  = 2,
#         ptour                  = 0.8,
#         interSampleWeight      = 1.0,
#         ae_power               = 2,
#         periodic_ae            = false,
#         rng                    = TaskLocalRNG()
#     ),
#     particleswarm         => (
#         n_particles            = 3,
#         w                      = 1.0,
#         c1                     = 2.0,
#         c2                     = 2.0,
#         prob_shift             = 0.25,
#         rng                    = TaskLocalRNG()
#     ),
#     adaptiveparticleswarm => (
#         n_particles            = 3,
#         c1                     = 2.0,
#         c2                     = 2.0,
#         prob_shift             = 0.25,
#         rng                    = TaskLocalRNG()
#     )
# )

# const TUNING_PARAMS = Dict{DataType,NamedTuple}(
#     AbstractClassification => (;
#         resampling              = Holdout(),
#         measure                 = LogLoss(tol = 2.22045e-16),
#         weights                 = nothing,
#         class_weights           = nothing,
#         repeats                 = 1,
#         operation               = nothing,
#         selection_heuristic     = MLJ.MLJTuning.NaiveSelection(nothing),
#         n                       = 25,
#         train_best              = true,
#         acceleration            = default_resource(),
#         acceleration_resampling = CPU1(),
#         check_measure           = true,
#         cache                   = true,
#     ),
#     AbstractRegression => (;
#         resampling              = Holdout(),
#         measure                 = MLJ.RootMeanSquaredError(),
#         weights                 = nothing,
#         class_weights           = nothing,
#         repeats                 = 1,
#         operation               = nothing,
#         selection_heuristic     = MLJ.MLJTuning.NaiveSelection(nothing),
#         n                       = 25,
#         train_best              = true,
#         acceleration            = default_resource(),
#         acceleration_resampling = CPU1(),
#         check_measure           = true,
#         cache                   = true,
#     ),
# )

# function range(
#     field  :: Union{Expr, Symbol};
#     lower  :: Union{AbstractFloat, Int, Nothing} = nothing,
#     upper  :: Union{AbstractFloat, Int, Nothing} = nothing,
#     origin :: Union{AbstractFloat, Int, Nothing} = nothing,
#     unit   :: Union{AbstractFloat, Int, Nothing} = nothing,
#     scale  :: OptSymbol             = nothing,
#     values :: Union{AbstractVector, Nothing}     = nothing,
# )
#     return function(model)
#         MLJ.range(
#             model,
#             field;
#             lower  = lower,
#             upper  = upper,
#             origin = origin,
#             unit   = unit,
#             scale  = scale,
#             values = values,
#         )
#     end
# end
