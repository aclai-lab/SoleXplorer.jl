# SoleXplorer - A symbolic journey through your datasets

[![CI](https://github.com/aclai-lab/SoleXplorer.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/aclai-lab/SoleXplorer.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/aclai-lab/SoleXplorer.jl/graph/badge.svg?token=EJQ1MJOTDO)](https://codecov.io/gh/aclai-lab/SoleXplorer.jl)

**SoleXplorer.jl** is an interactive interface for exploring symbolic machine learning models, built on top of the [Sole.jl](https://github.com/aclai-lab/Sole.jl) ecosystem. It provides tools for visualizing, inspecting, and interacting with models derived from (logic-based) symbolic learning algorithms.

---

## Installation

```julia
using Pkg
Pkg.add("https://github.com/aclai-lab/SoleXplorer.jl/")
```

---

## Quick Start

### Decision tree

```julia
using SoleXplorer

# Load example dataset
dataset = load_sole_dataset("iris")

# Train a decision tree (uses `DecisionTree.jl`)
model = fit_tree(dataset, :species)

# Explore the model
explorer = Explorer(model, dataset)
run(explorer)
```

### Temporal association rules

```julia
using SoleXplorer

# Load a temporal dataset
dataset = load_sole_dataset("temporal_events")

# Train a modal association ruleset (uses `ModalAssociationRules.jl`)
model = fit_modal_association_rules(dataset, :event_label; support=0.1, confidence=0.8)

# Start exploration
explorer = Explorer(model, logiset)
run(explorer)
```

In the exploration sessions, you can:
* View rules and their metrics
* Filter by modal depth or conditions
* Inspect logical formulas and their evaluation

---

## Under the Hood: How It Works

1. **Feature extraction**
   Data instances are seen as models of logical formalisms (see [`SoleLogics.jl`](https://github.com/aclai-lab/SoleLogics.jl)), and are represented in an optimized form for model checking (i.e., as _logisets_, see [`SoleData.jl`](https://github.com/aclai-lab/SoleData.jl)).

2. **Model Fitting**
   Symbolic models (e.g. decision trees, modal association rules) are trained via [`SoleModels.jl`](https://github.com/aclai-lab/SoleModels.jl)-compliant packages.

4. **Post-hoc Analysis**
   Metrics and human-readable logic representations are obtained via rule extraction algorithms from [`SolePostHoc.jl`](https://github.com/aclai-lab/SolePostHoc.jl).

5. **Association Analysis**
   Extract the association rules hidden in data via mining algorithms from [`ModalAssociationRules.jl`](https://github.com/aclai-lab/ModalAssociationRules.jl).

6. **Exploration Interface**
   `SoleXplorer.jl` ties everything into an interactive exploration session, allowing you to inspect rules, formulas, and patterns from your symbolic datasets and models.

---

## About

The package is developed by the [ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.

