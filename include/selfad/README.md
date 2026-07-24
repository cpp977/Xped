# Self-contained AD engine for Xped

This directory makes Xped's automatic differentiation **self-contained**: it is
a clean-room, header-only implementation of the small subset of the
[stan-math](https://github.com/stan-dev/math) reverse-mode API that Xped
actually uses. It keeps the `stan` / `stan::math` namespaces and the original
include paths (`stan/math/rev.hpp`, `stan/math/rev/core/vari.hpp`, ...), so
**no other file in Xped changes** — but building Xped with AD no longer
requires installing stan-math, TBB, SUNDIALS, or any of their transitive
dependencies.

## Why this works

Xped was already doing all the heavy AD lifting itself: every tensor-level
adjoint rule (contraction, permutation, SVD, truncation, ...) is hand-written
in `include/Xped/AD/ADTensor.hpp`, and the tensor tape node is Xped's own
`vari_value<Tensor>` specialization (`include/Xped/AD/vari_value.hpp`).
stan-math only supplied the *tape machinery* underneath — roughly a dozen
classes and functions. This directory reimplements exactly that machinery
(~800 lines, `stan/math/selfad_core.hpp`):

| Component | Purpose |
|---|---|
| `stack_alloc` | bump-pointer arena for tape nodes |
| `vari_base` / `chainable_alloc` | node interface / destructor registry for heap-owning nodes |
| `ChainableStack` | the global tape (`var_stack_`, `var_nochain_stack_`, `var_alloc_stack_`, `memalloc_`) |
| `vari_value<T>` / `var_value<T>` (floating point) | scalar value/adjoint nodes, `stan::math::var` |
| scalar operators, `sqrt`, `exp`, `log`, `pow`, ... | the `var` arithmetic used in energy expressions |
| `reverse_pass_callback`, `make_callback_var`, `callback_vari` | closure-based tape nodes |
| `nested_rev_autodiff`, `start_nested`, `recover_nested`, `grad()` | nested tapes; `grad()` sweeps only the innermost nested segment — this is what makes Xped's CTM checkpointing (nested re-forward inside an outer reverse sweep) work |
| `arena_allocator<T>` | STL allocator on the arena (`StanArenaPolicy`) |
| `finite_diff_gradient_auto` | 6th-order central differences for gradient checks |
| `stan/math/prim/eigen_plugins.h` | empty stand-in (Xped does AD per-tensor, never with Eigen matrices of `var`) |

## Build

Nothing to do — this is the default. `XPED_USE_AD=ON` now uses this engine and
pulls no AD-related external packages.

To compare against the original external backend:

```
cmake -DXPED_USE_EXTERNAL_STAN=ON ...        # plus vcpkg feature "external-stan"
```

which restores the previous behavior (stan-math + TBB + SUNDIALS via vcpkg).

## Verification

Two standalone test programs (no build system required):

* `tests/selfad_engine_tests.cpp` — engine-level: scalar/complex gradients vs
  analytic results, Xped's real `complex_var.hpp`/`reverse_pass_callback_alloc.hpp`
  compiled verbatim, a heap-owning tensor vari with leak counting, the
  checkpointing pattern, arena allocator, finite differences.

  ```
  g++ -std=c++20 -I include -I include/selfad tests/selfad_engine_tests.cpp -o selfad_tests && ./selfad_tests
  ```

* `tests/selfad_xped_tensor_test.cpp` — end-to-end on real Xped code:
  `Tensor<double,2,1,U1,AD=true>` pipelines (contract / adjoint / norm / trace
  and truncated SVD) differentiated on the self-contained tape, gradients
  validated against finite differences (agreement ~1e-8 relative). Needs
  Xped's ordinary header-only dependencies (Eigen, spdlog, fmt, yas, tabulate,
  seq, libassert, toml11, nlohmann-json, HighFive/HDF5, GSL, boost headers)
  but **no stan-math / TBB / SUNDIALS**.

Both were additionally run under AddressSanitizer + UBSan.

## Notes / semantics matched to stan-math

* `grad()` propagates only through the innermost nested segment
  (`beginning = empty_nested() ? 0 : size - nested_size()`), with index-based
  iteration so checkpoint callbacks may push/pop nodes mid-sweep.
* `vari_base::operator new` allocates from the arena; `chainable_alloc`
  registers for destruction at tape unwind. Xped's tensor varis inherit from
  both: the node lives in the arena while its heap-owned tensor storage is
  freed by the registered destructor (`delete` resolves to the no-op
  `vari_base::operator delete`).
* Single-threaded global tape (Xped uses AD single-threaded; matches
  stan-math built without `STAN_THREADS`).
