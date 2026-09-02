# Trading env — working notes

## PufferLib / Ocean basics

Ocean envs are vectorized, C-backed RL environments wired into Python via a
zero-copy numpy buffer layer. Each env directory has:

- `<env>.h` — the env struct, `c_reset`/`c_step`/`c_render`/`c_close`, and any
  env-specific logic. Header-only (no `.c` file compiled separately), so it's
  `#include`d directly by `binding.c`.
- `binding.c` — `#include "<env>.h"` + `../env_binding.h`, then implements
  `my_init` (reads scalar kwargs via `unpack()`) and `my_log` (exports `Log`
  fields to Python dicts). This is the actual Python C-extension entry point.
- `<env>.py` — the `PufferEnv` subclass. `single_observation_space` **must**
  be a `Box` and `single_action_space` **must** be `Discrete`/`MultiDiscrete`/
  `Box` (enforced in `pufferlib/pufferlib.py`) — no `Dict`/`Sequence`, because
  `env->observations` etc. are raw pointers into the same numpy array Python
  holds, assigned once in `env_binding.h`'s `vec_init` (no per-step copying).
- `<env>.c` (optional) — a standalone raylib demo/eval binary, built via
  `scripts/build_ocean.sh <env> local`, independent of `binding.c`/Python.

Variable-length data (e.g. however many options exist at a tick) has to be
padded to a fixed upper bound with a mask feature, since the observation
buffer's shape is fixed once at `vec_init` and can't vary per step.

## Trading env — current state

- Obs layout (flat float32, see `trading.py`): `MAX_OPTIONS` owned-option
  slots, then `MAX_OPTIONS` market-option slots, then `GLOBAL_FEATURES`
  scalars (cash, portfolio_value, tick, num_market, num_owned). Each option
  slot is `OPTION_FEATURES` floats: `[time_to_expiry, strike_price, price,
  type, mask]`.
- Action space: `MultiDiscrete([2]*MAX_OPTIONS)`, one buy/skip per market
  slot, index-aligned with the market block.
- `pufferlib/ocean/trading/utils.h` holds data structures/algorithms shared
  across the env: `Option`, `OwnedOption`, `OwnedOptionHeap` (min-heap by
  `expiry_tick`, so owned contracts aren't capped at `MAX_OPTIONS` the way
  market options are), and `load_options`/`store_options` (the only code that
  converts between typed `Option`s and the raw float buffer — `Option`'s
  `type`/`mask` fields aren't float-shaped, so they can't be pointer-cast
  onto `env->observations` directly).
- `c_step` flow: load market options from obs -> apply buy actions (pushing
  onto the owned heap) -> regenerate the market via `gen_test_options`
  (Black-Scholes pricing, calls only right now) -> pop+settle anything
  expired off the owned heap -> store market options and top-k owned back
  into obs.
- `trading.c` is currently a random-action smoke test (`BUY_PROBABILITY`
  chance per market slot per tick, no NN, no rendering/human control yet).

### Known open TODOs
- `time_to_expiry` is in years but is being used directly as a tick count
  for `expiry_tick` — needs a real ticks-per-year decision.
- Expiry settlement is a no-op — needs `underlying_price` tracked as real
  env state (only exists as a local param to `gen_test_options` right now).
- `binding.c` is still unmodified cartpole-era code (`my_init` unpacks
  `cart_mass`/`pole_mass`/etc., which don't exist on `Trading` anymore) — the
  Python extension won't build until this is brought in sync.
- `c_render` is stubbed out (`#warning`), no visual/human-playable demo yet.

## Formatting preferences for this env (given during this work)

- New data structures/algorithms go in `utils.h`, not `trading.h`.
- Comments: 2 lines max whenever possible. Prefer a clearer name/structure
  over a comment explaining an unclear one — code should be self-documenting
  first, commented second.
- Don't modify anything outside `pufferlib/ocean/trading/` (base PufferLib)
  without asking first.
- Check in before implementing several substantial new pieces in one
  unprompted pass — land one piece, or flag what's being bundled, rather
  than a large diff landing all at once.
