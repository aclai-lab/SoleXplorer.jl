# ---------------------------------------------------------------------------- #
#                          wolrds filters functions                            #
# ---------------------------------------------------------------------------- #
function fixed_windows(intervals::Vector{SoleLogics.Interval{Int64}}; nwindows::Int)
    i_filter = SoleLogics.IntervalLengthFilter(==, nwindows)
    collect(SoleLogics.filterworlds(i_filter, intervals))
end

function whole(intervals::Vector{SoleLogics.Interval{Int64}})
    max_upper = maximum(i.y for i in intervals)
    [SoleLogics.Interval{Int64}(1, max_upper)]
end

function absolute_movingwindow(intervals::Vector{SoleLogics.Interval{Int64}}; nwindows::Int, overlap::Int)
    """
    Generates a list of intervals from the input intervals that are of fixed size `nwindows` and overlap by a specified amount `overlap`.
    
    Args:
        intervals: A vector of `Interval{Int64}` representing the input intervals.
        nwindows: An integer specifying the fixed size for each interval.
        overlap: An integer specifying the overlap between consecutive intervals.
    
    Returns:
        A vector of `Interval{Int64}` containing intervals of size `nwindows` that properly overlap by `overlap`.
    """
    max_upper = maximum(i.y for i in intervals)
    last_win = max_upper - nwindows
    idx = 1

    worlds = SoleLogics.Interval{Int64}[]
    
    while idx <= length(intervals) && intervals[idx].x <= last_win
        if intervals[idx].y - intervals[idx].x == nwindows && (isempty(worlds) || worlds[end].y - overlap == intervals[idx].x)
            push!(worlds, intervals[idx])
        end
        idx += 1
    end

    return worlds
end

absolute_splitwindow(intervals::Vector{SoleLogics.Interval{Int64}}; nwindows::Int) = absolute_movingwindow(intervals; nwindows=nwindows, overlap=0)

function realtive_movingwindow(intervals::Vector{SoleLogics.Interval{Int64}}; nwindows::AbstractFloat, overlap::AbstractFloat)
    0.0 ≤ nwindows ≤ 1.0 || throw(ArgumentError("Window size ratio must be between 0.0 and 1.0"))
    0.0 ≤ overlap ≤ 1.0 || throw(ArgumentError("Overlap ratio must be between 0.0 and 1.0"))

    max_upper = maximum(i.y for i in intervals) -1
    absolute_window_size = floor(Int, max_upper * nwindows)
    absolute_overlap = floor(Int, absolute_window_size * overlap)
    
    absolute_movingwindow(intervals; nwindows=absolute_window_size, overlap=absolute_overlap)
end

function relative_splitwindow(intervals::Vector{SoleLogics.Interval{Int64}}; nwindows::AbstractFloat)
    0.0 ≤ nwindows ≤ 1.0 || throw(ArgumentError("Window size ratio must be between 0.0 and 1.0"))

    max_upper = maximum(i.y for i in intervals) -1
    absolute_movingwindow(intervals; nwindows=floor(Int, max_upper * nwindows), overlap=0)
end
