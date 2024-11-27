# ---------------------------------------------------------------------------- #
#                          wolrds filters functions                            #
# ---------------------------------------------------------------------------- #
function fixedlength_windows(intervals::Vector{SoleLogics.Interval{Int64}}; winsize::Int, kwargs...)
    i_filter = SoleLogics.IntervalLengthFilter(==, winsize)
    collect(SoleLogics.filterworlds(i_filter, intervals))
end

function whole(intervals::Vector{SoleLogics.Interval{Int64}}; kwargs...)
    max_upper = maximum(i.y for i in intervals)
    [SoleLogics.Interval{Int64}(1, max_upper)]
end

function absolute_movingwindow(intervals::Vector{SoleLogics.Interval{Int64}}; winsize::Int, overlap::Int, kwargs...)
    worlds = SoleLogics.Interval{Int64}[]

    step = winsize - overlap
    starts = collect(intervals[1].x:step:(intervals[end].y - winsize))

    for s in starts
        e = s + winsize
        push!(worlds, SoleLogics.Interval{Int64}(s, e))
    end

    return worlds
end

absolute_splitwindow(intervals::Vector{SoleLogics.Interval{Int64}}; winsize::Int, kwargs...) = absolute_movingwindow(intervals; winsize=winsize, overlap=0)

function relative_movingwindow(intervals::Vector{SoleLogics.Interval{Int64}}; winsize::AbstractFloat, overlap::AbstractFloat, kwargs...)
    0.0 ≤ winsize ≤ 1.0 || throw(ArgumentError("Window size ratio must be between 0.0 and 1.0"))
    0.0 ≤ overlap ≤ 1.0 || throw(ArgumentError("Overlap ratio must be between 0.0 and 1.0"))

    max_upper = maximum(i.y for i in intervals)
    absolute_window_size = floor(Int, max_upper * winsize)
    absolute_overlap = floor(Int, absolute_window_size * overlap)
    
    absolute_movingwindow(intervals; winsize=absolute_window_size, overlap=absolute_overlap)
end

function relative_splitwindow(intervals::Vector{SoleLogics.Interval{Int64}}; winsize::AbstractFloat, kwargs...)
    0.0 ≤ winsize ≤ 1.0 || throw(ArgumentError("Window size ratio must be between 0.0 and 1.0"))

    max_upper = maximum(i.y for i in intervals)
    absolute_movingwindow(intervals; winsize=floor(Int, max_upper * winsize), overlap=0)
end

function fixednumber_windows(intervals::Vector{SoleLogics.Interval{Int64}}; nwindows::Int, overlap::AbstractFloat, kwargs...)
    (nwindows ≥ 1) || throw(ArgumentError("Number of windows must be at least 1."))
    (0.0 ≤ overlap ≤ 1.0) || throw(ArgumentError("Overlap ratio must be between 0.0 and 1.0"))
    
    overall_start = first(intervals).x
    overall_end = last(intervals).y
    total_length = overall_end - overall_start
    
    p = clamp(overlap, 0.0, 1.0)
    denominator = nwindows * (1 - p)
    denominator > 0 || throw(ArgumentError("Invalid overlap percentage (overlap = $(p)) for nwindows = $(nwindows)"))
    
    window_size = max(round(Int, total_length / denominator), 1)
    step = round(Int, window_size * (1 - p))
    
    [SoleLogics.Interval{Int64}(s, min(s + window_size - 1, overall_end)) for s in overall_start:step:(overall_start + (nwindows-1)*step)]
end
