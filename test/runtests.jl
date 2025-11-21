using Test
using Random
using SemioticTransformer

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

    loss2, parts2 = SemioticTransformer.lossfn(model, tokens; λ_instab=1f-2, instab_samples=2)
    @test isfinite(loss2)
    @test isfinite(parts2.Linstab)
end
