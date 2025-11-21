# Semiotic Transformer

A compact Julia implementation of a "semiotic" Transformer that injects ideas from
structural linguistics, semiotics, and depth psychology into an otherwise minimal
Transformer block. The module lives under `src/SemioticTransformer.jl` in a
standard Julia package layout so it can still be read like a Karpathy-style
mini model while exposing
concepts such as Difference/Meaning fields, Greimas semiotic squares, and
multi-path meaning recomposition.

## Highlights

* **Noumenon latent (Kant)** – a tiny VAE-like encoder/decoder per block splits
  phenomena (`x`) from a latent `z`, with KL and reconstruction terms folded into
  the loss.
* **SelfField + coniunctio (Jung)** – a Self attractor plus a learnable Negation
  involution (`N^2≈I`) encourages merged opposites to sit near a potential
  optimum; the self loss uses `∇Φ(s)≈0` as a stationarity term.
* **Meaning/Difference fields** – the MeaningField provides a potential surface
  (and gradient) while the DifferenceField defines smooth quadratic distances
  used in attention and the Weber/JND constraint.
* **Will-flow (Schopenhauer)** – optional `will_step!` nudges representations via
  `+η∇Φ - ρ∇D`, letting the field dynamics push token states.
* **JND/Weber constraint (Fechner)** – `jnd_loss` calibrates the difference
  metric so small perturbations stay near a target detectability threshold.
* **SemioticSquare constraints** – metric regularisers over vocabulary slots keep
  A/¬A/B/¬B relationships coherent; multiple squares can be registered. Negation
  penalties are accumulated across blocks.
* **MeaningChainLayer** – separates denotation, connotation, and mythic
  pathways and recombines them with a learned softmax gate.
* **Batch- and padding-aware utilities** – forward and loss helpers accept both
  single sequences and column-stacked batches; supply `pad_token` to mask padded
  targets in the cross-entropy term while leaving semiotic penalties active.

## Quick start

1. Clone and activate the project (Flux `0.14`–`0.16` are supported):

   ```bash
   git clone https://github.com/you/semiotic-transformer.git
   cd semiotic-transformer
   julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
   ```

2. See something run immediately. The module file now self-hosts a tiny demo, so
   you can execute it directly:

   ```bash
   julia --project=. src/SemioticTransformer.jl
   ```

   You should see logging as the archetypal toy loop trains for a few dozen
   steps.

3. Run the same demo via a helper script that activates the project, honours an
   optional `SEMIOTIC_SEED`, and gives you a single command to copy/paste:

   ```bash
   SEMIOTIC_SEED=42 julia scripts/toy_train.jl
   ```

4. Import and call the same demo from the REPL or a script:

   ```julia
   julia --project=. -e 'using SemioticTransformer; SemioticTransformer.toy_train()'
   ```

If you prefer the explicit bootstrap that activates and instantiates the
project for you, the top-level `SemioticTransformer.jl` still performs that
step before loading the module. You can opt out with `SEMIOTIC_BOOTSTRAP=0`:

```julia
julia --project=. -e 'include("SemioticTransformer.jl"); using .SemioticTransformer'
```

Alternatively, run the helper script which activates the project, resolves, and
installs dependencies before returning control to the REPL:

```bash
julia scripts/instantiate.jl
julia --project=. -e 'include("SemioticTransformer.jl"); using .SemioticTransformer'
```

For a quick health check, a lightweight test now exercises the embedding,
forward pass, and loss helpers on a tiny synthetic batch:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

By default the toy example wires a SemioticSquare over the first four vocabulary
indices. The helper `lossfn(model, sequence)` will shift the sequence internally
(tokens `1:n-1` predict `2:n`), add the summed square loss across all registered
squares, accumulate Negation penalties across blocks, and apply the Kant/Jung/
Schopenhauer/Fechner terms (KL, reconstruction, Self, JND, Will-flow).

If your sequences contain padding, pass the pad token through to ignore those
positions during the cross-entropy term:

```julia
loss, parts = SemioticTransformer.lossfn(model, sequence; pad_token=0)
```

If you want manual control over the training pairs you can call the more explicit
method:

```julia
context, targets = SemioticTransformer.next_token_pairs(sequence)
loss, parts = SemioticTransformer.lossfn(model, context, targets; λ_square=0.05f0, λ_neg=0.01f0)
```

The loss helper now accepts `update_field` and `will` flags so you can decide
whether MeaningField prototypes are refreshed and the Will-flow step is applied
during the supervised forward pass (both default to `false` and `true`,
respectively):

```julia
loss, parts = SemioticTransformer.lossfn(model, sequence; update_field=true, will=false)
```

### Mini-batching

Every forward/loss helper also accepts a matrix of tokens whose columns represent
different sequences (shape `sequence_length × batch`). The `next_token_pairs`
utility and the loss function will shift each column independently and the model
returns logits shaped `classes × sequence_length × batch`.

```julia
sequences = hcat([1, 2, 5, 6], [2, 3, 4, 1])
loss, parts = SemioticTransformer.lossfn(model, sequences; λ_square=0.05f0)
```

When columns have padding (e.g. ragged sentences packed together), supply
`pad_token` so the masked positions are dropped from the cross-entropy term while
still flowing through the semiotic penalties:

```julia
sequences = hcat([1, 2, 5, 6], [2, 3, 0, 0])
loss, parts = SemioticTransformer.lossfn(model, sequences; pad_token=0)
```

To supervise more than one semiotic relation simultaneously, pass a collection of
`SemioticSquare`s via the `squares` keyword (or the legacy `square` for a single
constraint):

```julia
squares = [SemioticSquare(1, 2, 3, 4), SemioticSquare(5, 6, 7, 8)]
model = SemioticModel(vocab, d; squares=squares)
```

## Field dynamics and diagnostics

* **Negation stability** – the negation penalty combines involution, isometry,
  and symmetry terms; monitor `Lneg` from `lossfn` to keep `N^2≈I`.
* **Self anchoring** – the top block’s SelfField is pulled toward
  `coniunctio(x, ¬x)` for the first two tokens while enforcing `∇Φ(s)≈0`.
* **Will/field updates** – pass `update_field=true` or `will=true` to `forward`
  (as shown in `toy_train`) to let the MeaningField prototypes adapt or to apply
  the Will-flow step during the forward pass.
* **JND calibration** – tune `λ_jnd` and the `(k, θ)` pair inside `jnd_loss` so
  small perturbations are neither collapsed nor exaggerated by the difference
  metric.

## Extra-semiotic meaning: instability of the meaning field

“Meaning outside meaning” shows up as a change in interpretation without any
symbolic change. In this model that maps to **how sensitive the meaning field’s
gradient is to tiny, non-symbolic perturbations**. We can instrument that
directly:

* The helper `meaning_instability(mf, X; ε, samples)` measures how much the
  meaning gradient `∇Φ` jumps when every token embedding is nudged by noise
  `ε·N(0, I)`. Formally,

  ```
  Δ∇Φ = ∇Φ(x + ε) - ∇Φ(x)
  L_extra = mean‖Δ∇Φ‖²
  ```

* Wire this into the loss as a “meaning-instability penalty” to encourage
  robustness to non-symbolic jolts while keeping symbolic structure intact:

  ```julia
  loss, parts = SemioticTransformer.lossfn(model, tokens;
      λ_instab=1f-2, ε_instab=1f-3, instab_samples=4)
  @info parts.Linstab  # ≈ extra-semiotic volatility
  ```

* For manual probes, pull the top block’s meaning field and state activations
  from `forward` and call the helper directly:

  ```julia
  logits, KL, recL, acts = SemioticTransformer.forward(model, tokens)
  L_extra = SemioticTransformer.meaning_instability(model.blocks[end].mf, acts;
      ε=5f-4, samples=8)
  ```

The penalty is zero by default; enable it when you want to monitor or suppress
extra-semiotic drift induced by silence, timing, or other non-symbolic cues.

## Archetypal subcategories (V4) inside `SemioticTransformer`

The archetypal "V4" variant now lives directly inside `SemioticTransformer`
as the `SemioticTransformer.Archetypal` submodule. Each archetype is treated as
its own small category (local space `d_sub`) with dedicated
Negation/Meaning/Difference fields, a SemioticMHA, and a SelfField. A functor
(`U`, `V`) lifts each local morphism into the global space (`d`), and a router
produces per-token mixture weights across the archetypal units.

Key ideas:

* **Category + functor penalties** – `category_penalty` enforces associativity
  and identity on local morphisms; `functor_penalty` encourages the lifted
  morphisms to commute (`F(g∘f) ≈ F(g)∘F(f)`).
* **Archetype rules** – `archetype_rules_loss` ties together Self/Persona/
  Shadow/Anima/Animus/Trickster centers (after lifting to the global space) via
  coniunctio/negation relations and margin constraints.
* **Router** – `ArchetypeRouter` outputs soft assignments per token so the
  block behaves like a Mixture-of-Experts over archetypal subcategories.
* **Scene monoid (⊙)** – `SceneMonoid` projects each archetype’s local codes
  into a scene space `r`, mixes pairwise Hadamard products with router weights,
  and lifts the composed scene back to the global space. `λ_mono` controls the
  (soft) unit/associativity regulariser, while `λ_pair` scales the scene add-on
  in the block constructor. Use `allowed_pairs` in the archetypal constructors to
  prune the set of archetype interactions (default is all pairs) when you want
  to avoid the full `O(K^2)` scene mixing.
* **JND + Negation regularisers** – the global DifferenceField receives the
  same Weber-style `jnd_loss`, while local and global Negation operators are
  kept involutive/isometric via `negation_penalty`.

Minimal usage:

```julia
julia> include("SemioticTransformer.jl"); using .SemioticTransformer.Archetypal
julia> m = Archetypal.toy_train()
```

The archetypal helpers mirror the main API: `lossfn(model, sequence)` shifts
tokens internally, works with column-stacked batches, accepts `pad_token` to
mask padded targets, and lets you toggle field updates or will-flow via
`update_fields` / `will` keywords. For example:

```julia
seqs = hcat([1, 2, 5, 6], [2, 3, 0, 0])
loss, parts = Archetypal.lossfn(m, seqs; pad_token=0, λ_rules=1e-2, λ_struct=1e-3, λ_mono=1e-3)
```

You can tweak the archetypal geometry by adjusting the rule weight
(`λ_rules`), the category/functor weight (`λ_struct`), the monoid penalty
(`λ_mono`), the scene scale (`λ_pair`) and dimensionality (`r`), or the router
temperature (`τ`) inside `ArchetypeRouter` for harder or softer routing.
To trim the number of scene interactions, supply `allowed_pairs` to the
archetypal constructors:

```julia
pairs = [(1, 3), (2, 4), (1, 6)]  # Self×Shadow, Persona×Anima, Self×Trickster
m = Archetypal.ArchetypalModel(vocab, d; allowed_pairs=pairs, r=48)
```

## CI quick check

A minimal GitHub Actions workflow (`.github/workflows/ci.yml`) is provided to
instantiate the environment and ensure `SemioticTransformer.jl` loads cleanly
via `include`. This guards against syntax slips in the monolithic module even
when no standalone tests are present.

