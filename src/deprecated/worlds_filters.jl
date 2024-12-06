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
    step = winsize - overlap
    starts = collect(1:step:((intervals[end].y - winsize)))

    filter(intervals) do interval
        (interval.x in starts) && (interval.y - interval.x == winsize)
    end
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

function adaptive_moving_windows(intervals::Vector{SoleLogics.Interval{Int64}}; nwindows::Int, overlap::AbstractFloat=0.1, kwargs...)
    (nwindows ≥ 1) || throw(ArgumentError("Number of windows must be at least 1."))
    (0.0 ≤ overlap ≤ 1.0) || throw(ArgumentError("Overlap ratio must be between 0.0 and 1.0"))
    
    total_length = last(intervals).y - first(intervals).x
    total_length ≥ nwindows || throw(ArgumentError("Number of windows ($nwindows) is greater than total length ($total_length)"))
    
    winsize = round(Int, total_length / ((nwindows - 1) * (1 - overlap) + 1))
    ranges = SoleBase.movingwindow(total_length; nwindows=nwindows, window_size=winsize)

    collect(SoleLogics.Interval{Int64}(r[1],r[end]+1) for r in ranges[1:nwindows])
end
