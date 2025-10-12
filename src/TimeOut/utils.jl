# ---------------------------------------------------------------------------- #
#                               script builder                                 #
# ---------------------------------------------------------------------------- #
"""
    get_sph_algo(algo::Symbol)::String

Generate a Julia `using` statement for importing a specific algorithm from SolePostHoc.

This function creates a properly formatted import statement that will be injected into 
the `script_file` string when spawning a new Julia subprocess.

# Arguments
- `algo::Symbol`: The name of the algorithm/function to import from SolePostHoc

# Returns
- `String`: A formatted using statement in the form "using SolePostHoc: algo\\n"
"""
get_sph_algo(algo::Symbol)::String = "using SolePostHoc: $algo\n"

"""
    get_package(model::ValidModel)::String

Generate a Julia `using` statement for importing the package that defines the model type.

This function extracts the module name from a model object and creates a properly 
formatted import statement that will be injected into the `script_file` string when 
spawning a new Julia subprocess.

# Arguments
- `model::ValidModel`: A model object of a supported type (currently `DecisionTree.Ensemble`)

# Returns
- `String`: A formatted using statement in the form "using ModuleName\\n"

# Implementation Details
Uses `parentmodule(typeof(model))` to extract the module that defines the 
model's type.
"""
get_package(model::ValidModel)::String = "using $(parentmodule(typeof(model)))\n"

"""
    sph_script_builder(algo::Symbol, model::ValidModel; kwargs...)::String

Build a Julia script string for executing SolePostHoc algorithms in a subprocess.

This function is specific for the [`SolePostHoc.jl`](https://github.com/aclai-lab/SolePostHoc.jl) package.
It generates a complete Julia script that can be executed in a separate process to run
post-hoc explainability algorithms with timeout capabilities.

The Julia environment in which the process will be launched must have the dependent 
packages pre-installed. It is assumed that whoever uses this module will already have 
a properly configured environment with all the necessary dependencies.

# Arguments
- `algo::Symbol`: The SolePostHoc algorithm to execute (e.g., `:lumen`, `:batrees`)
- `model::ValidModel`: A model object of a supported type (currently `DecisionTree.Ensemble`)
- `kwargs...`: Optional keyword arguments to pass to the algorithm

# Returns
- `String`: A complete Julia script that:
  1. Imports necessary packages (JLD2, model package, SolePostHoc algorithm)
  2. Loads the model from a JLD2 file
  3. Executes the algorithm with provided kwargs
  4. Saves the result to a JLD2 file

# Examples
```julia
# Without kwargs
script = sph_script_builder(:lumen, model)

# With kwargs
script = sph_script_builder(:lumen, model; horizontal=0.2, max_depth=5)
```

# Generated Script Structure
The generated script follows this pattern:
```julia
using JLD2
using DecisionTree  # (or appropriate model package)
using SolePostHoc: lumen  # (or specified algorithm)
model = JLD2.load("\$model_file", "model")
result = lumen(model; horizontal=0.2)  # (with any provided kwargs)
JLD2.jldsave("\$result_file"; result=result)
```

# Necessary Dependencies
The target Julia environment must have these packages installed:
- [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl)
- [`SolePostHoc.jl`](https://github.com/aclai-lab/SolePostHoc.jl)
- [`DecisionTree.jl`](https://github.com/JuliaAI/DecisionTree.jl)
"""
function sph_script_builder(algo::Symbol, model::ValidModel; kwargs...)::String
    base_script  = "using JLD2\n"
    model_script = get_package(model)
    algo_model   = get_sph_algo(algo)
    kwargs_str   = isempty(kwargs) ? "" : join(["$k=$v" for (k, v) in pairs(kwargs)], ", ")

    execute = "model = JLD2.load(\"\$model_file\", \"model\")\nresult = $algo(model; $kwargs_str)\nJLD2.jldsave(\"\$result_file\"; result=result)\n"

    base_script * model_script * algo_model * execute
end

# ---------------------------------------------------------------------------- #
#                          working tmp directories                             #
# ---------------------------------------------------------------------------- #
# check and, if needed, convert a directory path to an absolute path
_abspath(dirname::String)::String = isabspath(dirname) ? dirname : joinpath(pwd(), dirname)

"""
    mk_tmp_dir(dirname::String="tmp")::String
    mk_tmp_dir(dirname::Symbol)::String

Create a temporary directory.

Throws an error if the directory already exists to prevent accidental overwrites.

# Arguments
- `dirname::String`: Name of the directory to create (default: "tmp")
- `dirname::Symbol`: Symbol version that gets converted to string

# Returns
- `String`: The path to the newly created directory

# Path Resolution
- If `dirname` is an absolute path, the directory will be created at that exact location
- If `dirname` is a relative path, the directory will be created in the current working directory
- Uses `pwd()` to determine the current working directory for relative paths
"""
function mk_tmp_dir(dirname::String="tmp")::String
    tmp_dir = _abspath(dirname)
    isdir(tmp_dir) ? throw(ArgumentError("Directory '$tmp_dir' already exists")) : mkdir(tmp_dir)
end
mk_tmp_dir(dirname::Symbol)::String = mk_tmp_dir(string(dirname))

"""
    rm_tmp_dir(dirname::String)::Bool
    rm_tmp_dir(dirname::Symbol)::Bool

Remove a temporary directory if it exists.

# Arguments
- `dirname::String`: Name of the directory to remove
- `dirname::Symbol`: Symbol version that gets converted to string

# Returns
- `Bool`: `true` if directory was removed, `false` if directory didn't exist

# Path Resolution
- If `dirname` is an absolute path, the directory will be removed from that exact location
- If `dirname` is a relative path, the directory will be removed from the current working directory
- Uses `pwd()` to determine the current working directory for relative paths
"""
function rm_tmp_dir(dirname::String)::Bool
    tmp_dir = _abspath(dirname)
    return if isdir(tmp_dir)
        rm(tmp_dir, recursive=true)
        true
    else
        false
    end
end
rm_tmp_dir(dirname::Symbol)::Bool = rm_tmp_dir(string(dirname))

# ---------------------------------------------------------------------------- #
#                              working tmp files                               #
# ---------------------------------------------------------------------------- #
"""
    tmp_file(extension::String="jld2")::String

Generate a temporary filename with the specified extension.

# Arguments
- `extension::String`: File extension to append (default: "jld2")

# Returns
- `String`: A unique temporary filename with the specified extension
"""
tmp_file(extension::String="jld2")::String = basename(tempname()) * "." * extension

"""
    abspath_tmp_file(tmp_dir::String, extension::String="jld2")::String

Generate an absolute path to a temporary file within a specified directory.

# Arguments
- `tmp_dir::String`: The directory path where the temporary file should be created
- `extension::String`: File extension for the temporary file (default: "jld2")

# Returns
- `String`: Complete absolute path to a unique temporary file
"""
abspath_tmp_file(tmp_dir::String, extension::String="jld2")::String = joinpath(tmp_dir, tmp_file(extension))