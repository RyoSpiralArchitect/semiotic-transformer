# Semiotic Transformer

A compact Julia implementation of a "semiotic" Transformer that injects ideas from
structural linguistics, semiotics, and depth psychology into an otherwise minimal
Transformer block. The module is kept in a single file (`SemioticTransformer.jl`)
so that it can be read like a Karpathy-style mini model while still exposing
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

The module does not ship with a package `Project.toml`; simply include the file
and use the module directly:

```julia
julia> include("SemioticTransformer.jl"); using .SemioticTransformer
```

You can then run the toy optimiser step:

```julia
julia> SemioticTransformer.toy_train()
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

