module TimeOut

# ---------------------------------------------------------------------------- #
#                                    models                                    #
# ---------------------------------------------------------------------------- #
using DecisionTree

"""
    ValidModel

A type union defining the supported model types for the TimeOut module.

This constant specifies which machine learning model types are compatible with
the [`SolePostHoc.jl`](https://github.com/aclai-lab/SolePostHoc.jl) timeout functionality provided by the `@run_with_timeout` macro.

# Supported Types
- `DecisionTree.Ensemble`: Random forest and other ensemble models from [`DecisionTree.jl`](https://github.com/JuliaAI/DecisionTree.jl)
"""
const ValidModel = Union{
    DecisionTree.Ensemble,
}
export ValidModel

# ---------------------------------------------------------------------------- #
#                                     utils                                    #
# ---------------------------------------------------------------------------- #
export get_sph_algo, get_package, sph_script_builder
export mk_tmp_dir, rm_tmp_dir
export tmp_file, abspath_tmp_file
include("utils.jl")

# ---------------------------------------------------------------------------- #
#                                     macro                                    #
# ---------------------------------------------------------------------------- #
macro run_with_timeout(algo, model, timeout_sec, kwargs)
    return quote
        local sph_expr  = $(string(expr))
        local sph_model = $(esc(model))
        local timeout   = $(esc(timeout_sec))
        local kwargs    = $(esc(kwargs))

        local tmp_dir     = mk_tmp_dir("timeout_tmp")
        local model_file  = abspath_tmp_file(tmp_dir)
        local result_file = abspath_tmp_file(tmp_dir)
        
        local result    = nothing
        local timed_out = false
        
        try
            # save the model to tmp file
            JLD2.jldsave(model_file; model=sph_model)
            
            # subprocess script
            local script_file = sph_script_builder(algo, model; kwargs...)
            
            # start the subprocess
            local proc = run(`julia -e $script_file`, wait=false)
            # for debug only
            # local proc = run(pipeline(`julia -e $script_file`, stdout=stdout, stderr=stderr), wait=false)
            
            # create timeout timer
            local timer = Timer(timeout) do t
                if process_running(proc)
                    @warn "Process exceeded timeout of $timeout seconds, killing..."
                    kill(proc, Base.SIGTERM)
                    sleep(0.5)
                    process_running(proc) && kill(proc, Base.SIGKILL)
                end
            end
            
            # wait for process to complete
            wait(proc)
            close(timer)
            
            # check if result file exists and load it
            if isfile(result_file)
                try
                    result = JLD2.load(result_file, "result")
                catch e
                    @warn "Error loading result: $e"
                    timed_out = true
                end
            else
                @warn "Result file not found, something went wrong in subprocess."
                timed_out = true
            end
            
        catch e
            @warn "Error in @run_with_timeout: $e"
            timed_out = true
        finally
            rm_tmp_dir(tmp_dir)
        end
        
        (result, timed_out)
    end
end

end