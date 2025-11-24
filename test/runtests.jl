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

    probe = SemioticTransformer.cognitive_probe(; vocab=vocab, d=d, seq=6, seed=7,
        λ_global=0.25f0, λ_couple=5f-3, λ_instab=1f-3, instab_samples=2, profile_width=6, λ_time=2f-3)
    @test isfinite(probe.Ltotal)
    @test probe.Lcouple >= 0
    @test isfinite(probe.local.Lce)
    @test isfinite(probe.global.Lce)
    @test probe.psi isa SemioticTransformer.PsiState
    @test isfinite(probe.instab)
    @test probe.L_time >= 0
    @test probe.flow !== nothing
    @test probe.dev_state isa SemioticTransformer.Archetypal.DevState
    @test !isempty(probe.spark)
    @test !isempty(probe.sweep)
    @test !isempty(probe.heatmaps.potential)
    @test !isempty(probe.coupling.spark)
    @test size(probe.coupling.matrix) == (global_K, local_k)
    @test length(probe.coupling.proto_min) == local_k
    @test length(probe.coupling.center_min) == global_K
    @test !isempty(probe.heatmaps.coupling)

    trace = SemioticTransformer.cognitive_trace(; vocab=vocab, d=d, seq=5, steps=3, seed=9,
        λ_global=0.25f0, λ_couple=5f-3, λ_instab=1f-3, instab_samples=2, profile_width=6, λ_time=2f-3)
    @test length(trace.trace) == 3
    @test occursin("step,L_total", trace.table)
    @test trace.coupling.spark != ""
    @test trace.coupling.heatmap != ""
    @test trace.psi isa SemioticTransformer.PsiState
    @test trace.dev_state isa SemioticTransformer.Archetypal.DevState
    @test all(row -> isfinite(row.L_total) && isfinite(row.instab), trace.trace)
    tmptrace = tempname()
    SemioticTransformer.save_cognitive_trace(tmptrace, trace.trace)
    @test isfile(tmptrace)
end

@testset "Archetypal time dynamics" begin
    Random.seed!(7)
    vocab = 8
    d = 24
    model = SemioticTransformer.Archetypal.ArchetypalModel(vocab, d; K=3, ds=12, r=8)

    tokens = rand(1:vocab, 6)
    targets = rand(1:vocab, 6)
    dev = SemioticTransformer.Archetypal.DevState()

    logits, Y, W, cache = SemioticTransformer.Archetypal.forward(model, tokens)
    L_time, flow = SemioticTransformer.Archetypal.time_dynamics_loss(model.block, Y, W, dev; λ_time=1f-2)
    @test isfinite(L_time)
    @test flow.H_start >= 0
    @test flow.conflict >= 0

    L_struct, parts_struct = SemioticTransformer.Archetypal.lossfn(model, tokens, targets)
    L_total, info = SemioticTransformer.Archetypal.total_loss(model, tokens, targets, dev; λ_time=1f-2)
    @test isfinite(L_struct)
    @test isfinite(L_total)
    SemioticTransformer.Archetypal.update!(dev, info.parts; flow=info.flow)
    @test 0f0 <= dev.m <= 1f0
    @test 0f0 <= dev.s <= 1f0

    L_base, _ = SemioticTransformer.Archetypal.base_loss(model, tokens, targets)
    @test isfinite(L_base)

    _, dev_state, trace = SemioticTransformer.Archetypal.train_with_time(; seed=9, steps=5, λ_time=1f-2, trace=true)
    @test 0f0 <= dev_state.m <= 1f0
    @test 0f0 <= dev_state.s <= 1f0
    @test length(trace) == 5

    table = SemioticTransformer.Archetypal.devstate_trace_table(trace)
    @test occursin("step,L_total", table)
    tmptrace = tempname()
    SemioticTransformer.Archetypal.save_devstate_trace(tmptrace, trace)
    @test isfile(tmptrace)
end
