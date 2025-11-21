using Test
using Random
using SemioticTransformer
using Flux

@testset "SemioticTransformer basics" begin
    Random.seed!(1234)
    vocab = 10
    d = 32
    model = SemioticTransformer.SemioticModel(vocab, d)

    tokens = rand(1:vocab, 8)
    logits, KL, recL, acts = SemioticTransformer.forward(model, tokens)
    @test size(logits) == (vocab, length(tokens))
    @test isfinite(KL)
    @test isfinite(recL)
    @test size(acts) == (d, length(tokens))

    loss, parts = SemioticTransformer.lossfn(model, tokens)
    @test isfinite(loss)
    @test all(isfinite, values(parts))

    instab = SemioticTransformer.meaning_instability(model.blocks[end].mf, acts; ε=1f-3, samples=2)
    @test instab >= 0

    sweep = SemioticTransformer.meaning_instability_profile(model.blocks[end].mf, acts; epsilons=[1f-4, 1f-3], samples=2)
    @test length(sweep) == 2
    @test all(v -> v[2] >= 0 && isfinite(v[2]), sweep)
    spark = SemioticTransformer.instability_sparkline(sweep; width=4)
    @test !isempty(spark)
    tmp = tempname()
    SemioticTransformer.save_instability_profile(tmp, sweep)
    @test isfile(tmp)
    lines = readlines(tmp)
    @test length(lines) == length(sweep) + 1

    heat = SemioticTransformer.ascii_heatmap(rand(Float32, 2, 3))
    @test !isempty(heat)

    loss2, parts2 = SemioticTransformer.lossfn(model, tokens; λ_instab=1f-2, instab_samples=2)
    @test isfinite(loss2)
    @test isfinite(parts2.Linstab)

    probe = SemioticTransformer.instability_probe(; samples=2, λ_instab=1f-2)
    @test isfinite(probe.instab)
    @test isfinite(probe.parts.Linstab)
    @test length(probe.sweep) >= 1
    @test !isempty(probe.heatmaps.potential)
    @test !isempty(probe.heatmaps.difference)
end

@testset "Cognitive bridge" begin
    Random.seed!(2024)
    vocab = 9
    d = 24
    emb = Flux.Embedding(vocab, d)
    cog = SemioticTransformer.CognitiveModel(emb; local_layers=1, local_k=4, local_z=12, global_K=3, global_ds=12, global_r=16)

    tokens = rand(1:vocab, 7)
    context, targets = SemioticTransformer.next_token_pairs(tokens)

    out = SemioticTransformer.forward(cog, context)
    @test size(out.local.logits) == (vocab, length(context))
    @test out.psi isa SemioticTransformer.PsiState
    @test out.psi.X === out.local.acts
    @test cog.emb === cog.local.emb === cog.global.emb

    loss, parts = SemioticTransformer.lossfn(cog, context, targets; λ_couple=1f-3, λ_global=0.5f0)
    @test isfinite(loss)
    @test isfinite(parts.local.Lce)
    @test isfinite(parts.global.Lce)
    @test parts.Lcouple >= 0

    cpen = SemioticTransformer.coupling_penalty(cog.local, cog.global)
    @test cpen >= 0
end
