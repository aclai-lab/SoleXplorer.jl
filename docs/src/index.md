![header](https://raw.githubusercontent.com/aclai-lab/SoleXplorer.jl/refs/heads/main/logo.png)

A **library** for symbolic learning, timeseries analysis and rules extraction

# Introduction

**SoleXplorer** is an interactive interface for exploring symbolic machine learning models, built on top of the [Sole.jl](https://github.com/aclai-lab/Sole.jl) ecosystem. It provides tools for visualizing, inspecting, and interacting with models derived from (logic-based) symbolic learning algorithms.
Built upon [MLJ framework](https://juliaai.github.io/MLJ.jl/stable/), extending its funcionality with tools like the ability of treat **time-series** analysis or the ability to retrieve what **rules** was used and how are interconnected.
It also offers the user the ability to build a machine-learining pipeline using only a function call, as the following code snippet demonstrates.

```julia
using SoleXplorer, MLJ

Xc, yc = @load_iris

modelc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
```

## What is Symbolic Logic?

## Installation

You can install SoleXplorer by typing the following in the Julia REPL:
```julia
using Pkg
Pkg.add("https://github.com/aclai-lab/SoleXplorer.jl/")
```

followed by 
```julia
using SoleXplorer
```
to load the package.

## Overview
First, let's load a dataset:
```julia
using MLJ, DataFrames

Xc, yc = @load_iris
Xc = DataFrame(Xc)
```

SoleXplorer operates through 3 high-level functions, designed to be used sequentially:

**setup_dataset()**: prepares the dataset for analysis, where you set the parameters needed for dataset formatting and the related MLJ machine.
- choose the model to use.
- decide the resampling (or cross validation) strategy, with related train, validation and test ratio values.
- decide the tuning strategy to apply.
- set the global RNG seed.
- For time series datasets, it's possible to set features, windows for the type of compression suitable for the chosen model.

```julia
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)    
)
```

**train_test()**: is the engine of SoleXplorer, built on top of MLJ and Sole:
- trains the machine through MLJ.
- converts the result into a Logic structure through SoleModels.
- tests the model.

```julia
solemc = train_test(dsc)
```

**symbolic_analysis()**: extracts the requested information
- we can specify the type and related parameters for rule extraction.
- we can request standard analysis metrics (accuracy, rms, ...).

```julia
modelc = symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)
```

But it's also possible to condense the various parameters into a single call to symbolic_analysis:

```julia
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2),
    extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
```

One of the peculiarities of SoleXplorer is being as automated as possible, to make it easily usable even for machine learning newcomers.
The simplest way to launch an analysis with SoleXplorer is:

```julia
modelc = symbolic_analysis(Xc, yc)
```



<!-- To start out, let's discuss the high-level functionality provided by the package, which hopefully will help direct you to more specific documentation for your use-case:

  * [`CSV.File`](@ref): the most commonly used function for ingesting delimited data; will read an entire data input or vector of data inputs, detecting number of columns and rows, along with the type of data for each column. Returns a `CSV.File` object, which is like a lightweight table/DataFrame. Assuming `file` is a variable of a `CSV.File` object, individual columns can be accessed like `file.col1`, `file[:col1]`, or `file["col"]`. You can see parsed column names via `file.names`. A `CSV.File` can also be iterated, where a `CSV.Row` is produced on each iteration, which allows access to each value in the row via `row.col1`, `row[:col1]`, or `row[1]`. You can also index a `CSV.File` directly, like `file[1]` to return the entire `CSV.Row` at the provided index/row number. Multiple threads will be used while parsing the input data if the input is large enough, and full return column buffers to hold the parsed data will be allocated. `CSV.File` satisfies the [Tables.jl](https://github.com/JuliaData/Tables.jl) "source" interface, and so can be passed to valid sink functions like `DataFrame`, `SQLite.load!`, `Arrow.write`, etc. Supports a number of keyword arguments to control parsing, column type, and other file metadata options.
  * [`CSV.read`](@ref): a convenience function identical to `CSV.File`, but used when a `CSV.File` will be passed directly to a sink function, like a `DataFrame`. In some cases, sinks may make copies of incoming data for their own safety; by calling `CSV.read(file, DataFrame)`, no copies of the parsed `CSV.File` will be made, and the `DataFrame` will take direct ownership of the `CSV.File`'s columns, which is more efficient than doing `CSV.File(file) |> DataFrame` which will result in an extra copy of each column being made. Keyword arguments are identical to `CSV.File`. Any valid Tables.jl sink function/table type can be passed as the 2nd argument. Like `CSV.File`, a vector of data inputs can be passed as the 1st argument, which will result in a single "long" table of all the inputs vertically concatenated. Each input must have identical schemas (column names and types).
  * [`CSV.Rows`](@ref): an alternative approach for consuming delimited data, where the input is only consumed one row at a time, which allows "streaming" the data with a lower memory footprint than `CSV.File`. Supports many of the same options as `CSV.File`, except column type handling is a little different. By default, every column type will be essentially `Union{Missing, String}`, i.e. no automatic type detection is done, but column types can be provided manually. Multithreading is not used while parsing. After constructing a `CSV.Rows` object, rows can be "streamed" by iterating, where each iteration produces a `CSV.Row2` object, which operates similar to `CSV.File`'s `CSV.Row` type where individual row values can be accessed via `row.col1`, `row[:col1]`, or `row[1]`. If each row is processed individually, additional memory can be saved by passing `reusebuffer=true`, which means a single buffer will be allocated to hold the values of only the currently iterated row. `CSV.Rows` also supports the Tables.jl interface and can also be passed to valid sink functions.
  * [`CSV.Chunks`](@ref): similar to `CSV.File`, but allows passing a `ntasks::Integer` keyword argument which will cause the input file to be "chunked" up into `ntasks` number of chunks. After constructing a `CSV.Chunks` object, each iteration of the object will return a `CSV.File` of the next parsed chunk. Useful for processing extremely large files in "chunks". Because each iterated element is a valid Tables.jl "source", `CSV.Chunks` satisfies the `Tables.partitions` interface, so sinks that can process input partitions can operate by passing `CSV.Chunks` as the "source".
  * [`CSV.write`](@ref): A valid Tables.jl "sink" function for writing any valid input table out in a delimited text format. Supports many options for controlling the output like delimiter, quote characters, etc. Writes data to an internal buffer, which is flushed out when full, buffer size is configurable. Also supports writing out partitioned inputs as separate output files, one file per input partition. To write out a `DataFrame`, for example, it's simply `CSV.write("data.csv", df)`, or to write out a matrix, it's `using Tables; CSV.write("data.csv", Tables.table(mat))`
  * [`CSV.RowWriter`](@ref): An alternative way to produce csv output; takes any valid Tables.jl input, and on each iteration, produces a single csv-formatted string from the input table's row.

That's quite a bit! Let's boil down a TL;DR:
  * Just want to read a delimited file or collection of files and do basic stuff with data? Use [`CSV.File(file)`](@ref CSV.File) or [`CSV.read(file, DataFrame)`](@ref CSV.read)
  * Don't need the data as a whole or want to stream through a large file row-by-row? Use [`CSV.Rows`](@ref).
  * Want to process a large file in "batches"/chunks? Use [`CSV.Chunks`](@ref).
  * Need to produce a csv? Use [`CSV.write`](@ref).
  * Want to iterate an input table and produce a single csv string per row? [`CSV.RowWriter`](@ref).

For the rest of the manual, we're going to have two big sections, *[Reading](@ref)* and *[Writing](@ref)* where we'll walk through the various options to `CSV.File`/`CSV.read`/`CSV.Rows`/`CSV.Chunks` and `CSV.write`/`CSV.RowWriter`.

```@contents
Pages = ["reading.md", "writing.md", "examples.md"]
``` -->