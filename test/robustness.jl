using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ, DataFrames, Random
using CategoricalArrays, JLD2

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SX.NatopsLoader()
Xts, yts = SX.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                    decisiontree classification robustness                    #
# ---------------------------------------------------------------------------- #
@testset "decisiontree classification data validation" begin
    for fraction_train in 0.5:0.1:0.9
        for seed in 1:5:40
            for min_purity_increase in 0.0:0.1:0.3
                for max_depth in 1:10
                    model = symbolic_analysis(
                        Xc, yc;
                        model=DecisionTreeClassifier(;max_depth, min_purity_increase),
                        resampling=Holdout(; fraction_train, shuffle=true),
                        seed,
                        measures=(accuracy,)
                    )
                    sx_acc = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(model.ds.mach, model.ds.mach.args[1].data[model.ds.pidxs[1].test, :])
                    mlj_acc = accuracy(yhat, model.ds.mach.args[2].data[model.ds.pidxs[1].test])

                    @test sx_acc == mlj_acc
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                      decisiontree regression robustness                      #
# ---------------------------------------------------------------------------- #
@testset "decisiontree regression data validation" begin
    for fraction_train in 0.5:0.1:0.9
        for seed in 1:5:40
            for min_purity_increase in 0.0:0.1:0.3
                for max_depth in 1:10
                    model = symbolic_analysis(
                        Xr, yr;
                        model=DecisionTreeRegressor(;max_depth, min_purity_increase),
                        resampling=Holdout(; fraction_train, shuffle=true),
                        seed,
                        measures=(rms,)
                    )
                    sx_rms = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(model.ds.mach, model.ds.mach.args[1].data[model.ds.pidxs[1].test, :])
                    mlj_rms = rms(yhat, model.ds.mach.args[2].data[model.ds.pidxs[1].test])

                    @test sx_rms == mlj_rms
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                    randomforest classification robustness                    #
# ---------------------------------------------------------------------------- #
@testset "randomforest classification data validation" begin
    for fraction_train in 0.5:0.1:0.9
        for seed in 1:5:40
            for sampling_fraction in 0.7:0.1:0.9
                for n_trees in 10:10:100
                    model = symbolic_analysis(
                        Xc, yc;
                        model=RandomForestClassifier(;n_trees, sampling_fraction),
                        resampling=Holdout(; fraction_train, shuffle=true),
                        seed,
                        measures=(accuracy,)
                    )
                    sx_acc = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(model.ds.mach, model.ds.mach.args[1].data[model.ds.pidxs[1].test, :])
                    mlj_acc = accuracy(yhat, model.ds.mach.args[2].data[model.ds.pidxs[1].test])

                    @test sx_acc == mlj_acc
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                      randomforest regression robustness                      #
# ---------------------------------------------------------------------------- #
@testset "randomforest regression data validation" begin
    for fraction_train in 0.5:0.1:0.9
        for seed in 1:5:40
            for sampling_fraction in 0.7:0.1:0.9
                for n_trees in 10:10:100
                    model = symbolic_analysis(
                        Xr, yr;
                        model=RandomForestRegressor(;n_trees, sampling_fraction),
                        resampling=Holdout(; fraction_train, shuffle=true),
                        seed,
                        measures=(rms,)
                    )
                    sx_rms = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(model.ds.mach, model.ds.mach.args[1].data[model.ds.pidxs[1].test, :])
                    mlj_rms = rms(yhat, model.ds.mach.args[2].data[model.ds.pidxs[1].test])

                    @test sx_rms == mlj_rms
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                              adaboost robustness                             #
# ---------------------------------------------------------------------------- #
@testset "adaboost classification data validation" begin
    for fraction_train in 0.5:0.1:0.9
        for seed in 1:40
            for feature_importance in [:impurity, :split]
                for n_iter in 1:5:100
                    model = symbolic_analysis(
                        Xc, yc;
                        model=AdaBoostStumpClassifier(;n_iter, feature_importance),
                        resampling=Holdout(; fraction_train=0.7, shuffle=true),
                        seed,
                        measures=(accuracy,)
                    )
                    sx_acc = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(model.ds.mach, model.ds.mach.args[1].data[model.ds.pidxs[1].test, :])
                    mlj_acc = accuracy(yhat, model.ds.mach.args[2].data[model.ds.pidxs[1].test])

                    @test sx_acc == mlj_acc
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                  xgboost binary classification robustness                    #
# ---------------------------------------------------------------------------- #
data_path = joinpath(@__DIR__, "juliacon2025/respiratory_pneumonia.jld2")
data  = JLD2.load(data_path)
Xb = data["X"]
yb = CategoricalArrays.CategoricalArray{String,1,UInt32}(data["y"])

@testset "xgboost binary classification data validation" begin
    for fraction_train in 0.5:0.1:0.9
        for seed in 1:5:40
            for num_round in 10:5:50
                for eta in 0.1:0.1:0.4
                    model = symbolic_analysis(
                        Xb, yb;
                        model=XGBoostClassifier(;eta, num_round),
                        resampling=Holdout(; fraction_train, shuffle=true),
                        seed,
                        measures=(accuracy,)
                    )
                    sx_acc = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(model.ds.mach, model.ds.mach.args[1].data[model.ds.pidxs[1].test, :])
                    mlj_acc = accuracy(yhat, model.ds.mach.args[2].data[model.ds.pidxs[1].test])

                    @test sx_acc == mlj_acc
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                   xgboost multi classification robustness                    #
# ---------------------------------------------------------------------------- #
@testset "xgboost multi classification data validation" begin
    for fraction_train in 0.5:0.1:0.9
        for seed in 1:5:40
            for num_round in 10:5:50
                for eta in 0.1:0.1:0.4
                    model = symbolic_analysis(
                        Xc, yc;
                        model=XGBoostClassifier(;eta, num_round),
                        resampling=Holdout(; fraction_train, shuffle=true),
                        seed,
                        measures=(accuracy,)
                    )
                    sx_acc = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(model.ds.mach, model.ds.mach.args[1].data[model.ds.pidxs[1].test, :])
                    mlj_acc = accuracy(yhat, model.ds.mach.args[2].data[model.ds.pidxs[1].test])

                    @test sx_acc == mlj_acc
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                        xgboost regression robustness                         #
# ---------------------------------------------------------------------------- #
@testset "xgboost regression data validation" begin
    for fraction_train in 0.5:0.1:0.9
        for seed in 1:5:40
            for num_round in 10:5:50
                for eta in 0.1:0.1:0.4
                    model = symbolic_analysis(
                        Xr, yr;
                        model=XGBoostRegressor(;eta, num_round),
                        resampling=Holdout(; fraction_train, shuffle=true),
                        seed,
                        measures=(rms,)
                    )
                    sxhat = solemodels(model)[1].info.supporting_predictions
                    yhat = MLJ.predict_mode(model.ds.mach, model.ds.mach.args[1].data[model.ds.pidxs[1].test, :])
                    sx_rms = model.measures.measures_values[1]
                    mlj_rms = rms(yhat, model.ds.mach.args[2].data[model.ds.pidxs[1].test])

                    @test isapprox(sxhat, yhat; rtol=1e-6)
                    @test isapprox(sx_rms, mlj_rms; rtol=1e-6)
                end
            end
        end
    end
end
