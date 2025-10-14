using Test
using SoleXplorer

using MLJ, DataFrames
using DecisionTree

# ---------------------------------------------------------------------------- #
#                                test dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
model = DecisionTree.build_forest(yc, MLJ.matrix(Xc))

# ---------------------------------------------------------------------------- #
#                               script builder                                 #
# ---------------------------------------------------------------------------- #
# get_sph_algo
@test get_sph_algo(:lumen) == "using SolePostHoc: lumen\n"

# get_package
@test get_package(model) == "using DecisionTree\n"

@test sph_script_builder(:lumen, model) == "using JLD2\nusing DecisionTree\nusing SolePostHoc: lumen\nmodel = JLD2.load(\"\$model_file\", \"model\")\nresult = lumen(model; )\nJLD2.jldsave(\"\$result_file\"; result=result)\n"
@test sph_script_builder(:lumen, model; horizontal=0.2) == "using JLD2\nusing DecisionTree\nusing SolePostHoc: lumen\nmodel = JLD2.load(\"\$model_file\", \"model\")\nresult = lumen(model; horizontal=0.2)\nJLD2.jldsave(\"\$result_file\"; result=result)\n"

# ---------------------------------------------------------------------------- #
#                          working tmp directories                             #
# ---------------------------------------------------------------------------- #
# mk_tmp_dir
current_dir = pwd()
tmp_dir = mk_tmp_dir()
@test tmp_dir == current_dir * "/tmp"
@test_throws ArgumentError mk_tmp_dir()
tmp_dir = mk_tmp_dir(:symbol_tmp)
@test tmp_dir == current_dir * "/symbol_tmp"

# tmp_file
@test endswith(tmp_file(), ".jld2")
@test endswith(tmp_file("wav"), ".wav")

# Test abspath_tmp_file function
abs_file1 = abspath_tmp_file(tmp_dir)
abs_file2 = abspath_tmp_file(tmp_dir, "wav")
@test startswith(abs_file1, tmp_dir)
@test endswith(abs_file1, ".jld2")
@test startswith(abs_file2, tmp_dir)
@test endswith(abs_file2, ".wav")

# rm_tmp_dir
@test rm_tmp_dir("/invalid/invalid") == false
@test rm_tmp_dir(:invalid) == false
@test rm_tmp_dir("tmp") == true
@test rm_tmp_dir(:symbol_tmp) == true

