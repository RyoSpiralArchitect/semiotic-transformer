#!/usr/bin/env julia
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using SemioticTransformer

seed = parse(Int, get(ENV, "SEMIOTIC_SEED", "2025"))
steps = parse(Int, get(ENV, "SEMIOTIC_STEPS", "80"))
log_every = parse(Int, get(ENV, "SEMIOTIC_LOG_EVERY", "10"))
will_start = parse(Int, get(ENV, "SEMIOTIC_WILL_START", "20"))
field_start = parse(Int, get(ENV, "SEMIOTIC_FIELD_START", "40"))

parsebool(s::AbstractString; default::Bool=true) = lowercase(strip(s)) in ("1", "true", "t", "yes", "y", "on") ? true :
                                                  lowercase(strip(s)) in ("0", "false", "f", "no", "n", "off") ? false :
                                                  default

will = parsebool(get(ENV, "SEMIOTIC_WILL", "true"); default=true)
update_fields = parsebool(get(ENV, "SEMIOTIC_UPDATE_FIELDS", "true"); default=true)

@info "Launching Archetypal.toy_train" seed steps log_every will_start field_start will update_fields
SemioticTransformer.Archetypal.toy_train(;
    seed=seed,
    steps=steps,
    log_every=log_every,
    will_start=will_start,
    field_start=field_start,
    will=will,
    update_fields=update_fields,
)
