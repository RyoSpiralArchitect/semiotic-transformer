#!/usr/bin/env julia
# Resolves and instantiates the project environment before loading the model.
using Pkg
Pkg.activate(@__DIR__*"/..")
Pkg.resolve()
Pkg.instantiate()
