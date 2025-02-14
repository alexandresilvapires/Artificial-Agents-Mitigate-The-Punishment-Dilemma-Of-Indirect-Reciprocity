# Source code for "Artificial Agents Mitigate The Punishment Dilemma Of Indirect Reciprocity"

Accepted as a full paper for AAMAS 2025 (24th International Conference on Autonomous Agents and Multiagent Systems, https://aamas2025.org/). 

This folder contains all the source code to reproduce the results of the paper.

## Requirements

The code base is entirely done in Julia 1.10.

The required Julia packages can be added via the command

```
] add CairoMakie Colors LaTeXStrings NonlinearSolve Memoization ArnoldiMethod Serialization
```

The Julia code also makes use of the following standard libraries: LinearAlgebra, Statistics, Test, Dates, SparseArrays

## Code Structure

Most of the code files do the mathematical operations described in the paper, consisting of calculating the cooperation index, disagreement, gradient of selection and more for a given set of parameters. The structure of the code is as follows:

* **stats.jl** - The primary file used to run experiments. It contains all the parameters that the experiment uses, calls the relevant functions to obtain the metrics mentioned above, and calls the plotting functions;
* **reputation.jl** - Contains all the code to calculate the reputation equilibrium, via ODEs, at a given strategy state, as well as disagreement metrics at that state;
* **strategy.jl** - Contains all the code to calculate the full strategy Markov chain, using reputation.jl at each state to calculate transition probabilities. Also calculates all metrics at each state and packs them for stats.jl;
* **plotter.jl** - Contains all code relative to plotting for each of the type of experiment in stats.jl;
* **tests.jl** - Contains tests to verify the code base;
* **utils.jl** - Contains utility code used throughout the code base, pertaining to data processing and some common mathematical functions such as obtaining the stationary distribution of the strategy Markov chain;

Running the Julia code produces a folder in the "Plots/<foldername>" directory, where the data is then stored. This data is then read to make the plots. This data can then be re-read to remake plots without rerunning experiments.
