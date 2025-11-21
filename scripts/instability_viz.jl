using Pkg
Pkg.activate(@__DIR__ |> dirname)
using Printf
using SemioticTransformer

# Environment knobs
seed = get(ENV, "SEMIOTIC_SEED", "2027") |> x -> parse(Int, x)
ε = get(ENV, "SEMIOTIC_EPS", "5e-4") |> x -> parse(Float64, x)
samples = get(ENV, "SEMIOTIC_SAMPLES", "4") |> x -> parse(Int, x)
λ = get(ENV, "SEMIOTIC_LAMBDA_INSTAB", "1e-2") |> x -> parse(Float64, x)

data = SemioticTransformer.instability_probe(
    seed=seed, ε=ε, samples=samples, λ_instab=λ,
    visualize=true,
)

println("\nSweep (ε → instability):")
for (ϵ, val) in data.sweep
    @printf("  %8.2e : %0.6f\n", ϵ, val)
end

println("\nPotential heatmap (per token):")
println(data.heatmaps.potential)

println("\nDifference-field heatmap:")
println(data.heatmaps.difference)

println("\n‖∇Φ‖ heatmap (per token):")
println(data.heatmaps.grad_norms)
