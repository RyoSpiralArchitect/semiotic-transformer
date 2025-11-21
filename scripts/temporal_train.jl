#!/usr/bin/env julia
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using SemioticTransformer

steps = parse(Int, get(ENV, "SEMIOTIC_TIME_STEPS", "200"))
seed = parse(Int, get(ENV, "SEMIOTIC_SEED", "2025"))
λ_time = parse(Float64, get(ENV, "SEMIOTIC_TIME_LAMBDA", "1e-2"))
trace_path = get(ENV, "SEMIOTIC_DEV_TRACE", "")

model, dev_state, trace = SemioticTransformer.Archetypal.train_with_time(
    ; seed=seed, steps=steps, λ_time=Float32(λ_time), trace=true,
    save_trace=isempty(trace_path) ? nothing : trace_path,
)

last = trace[end]
println("Final DevState: m=$(dev_state.m), s=$(dev_state.s)")
println("Temporal losses: L_struct=$(last.L_struct), L_time=$(last.L_time)")
println("Trace length: $(length(trace))")

if !isempty(trace_path)
    println("Saved DevState trace to $trace_path")
end
