using Documenter
using SoleXplorer

# DocMeta.setdocmeta!(AudioReader, :DocTestSetup, :(using AudioReader); recursive = true)

# makedocs(;
#     modules=[AudioReader],
#     authors="Riccardo Pasini",
#     repo=Documenter.Remotes.GitHub("PasoStudio73", "AudioReader.jl"),
#     sitename="AudioReader.jl",
#     format=Documenter.HTML(;
#         size_threshold=4000000,
#         prettyurls=get(ENV, "CI", "false") == "true",
#         canonical="https://PasoStudio73.github.io/AudioReader.jl",
#         assets=String[],
#     ),
#     pages=[
#         "Home"      => "index.md",
#         "Reference" => "reference.md",
#     ],
#     warnonly=:true,
# )

# deploydocs(;
#     repo = "github.com/PasoStudio73/AudioReader.jl",
#     devbranch = "main",
#     target = "build",
#     branch = "gh-pages",
#     versions = ["main" => "main", "stable" => "v^", "v#.#"],
# )
