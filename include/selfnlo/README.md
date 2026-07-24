# Self-contained nonlinear optimizer for Xped

This directory removes Xped's dependency on **ceres-solver** (and its
transitive glog/gflags/SuiteSparse chain): it is a clean-room, header-only
implementation of the ceres `GradientProblemSolver` API subset that Xped's
`NLO/CeresSolve.hpp` uses. The `ceres` namespace and include paths
(`ceres/first_order_function.h`, `ceres/gradient_problem.h`,
`ceres/gradient_problem_solver.h`) are preserved, so **no other Xped file
changes**.

## Algorithm

The direction engine is a port of TeneT.c's LBFGS driver
(`src/optimize/lbfgs.cpp`), itself a documented, pinned port of
**OptimKit.jl v0.4.2**:

* L-BFGS two-loop recursion with normalized curvature pairs
  `(s/|s|, y/|s|, rho = <s,s>/<s,y>)`, initial Hessian scaling
  `gamma = <s,y>/<y,y>`, curvature-pair acceptance
  `<s,y>/<s,s> > |g|/10000`;
* soft failure degradation: a zero-step line search clears the history and
  retries with steepest descent before returning the best iterate seen;
* `NONLINEAR_CONJUGATE_GRADIENT` (FR / PR+ / HS, with Powell restarts and a
  tightened c2 = 0.1 curvature condition) and `STEEPEST_DESCENT` share the
  same line searches.

Two line-search engines back the `WOLFE` option, selected by the
`Options::line_search_engine` extension field:

* **`INTERPOLATING_WOLFE` (default, ceres-faithful)** — the classic
  bracket-and-zoom strong-Wolfe search (Nocedal & Wright, Algorithms
  3.5/3.6) with safeguarded polynomial interpolation, the same construction
  as ceres' WOLFE line search.  `Options::line_search_interpolation_type`
  selects the zoom model: `CUBIC` (Hermite interpolation on values and
  slopes, the ceres default), `QUADRATIC`, or `BISECTION`; a degenerate or
  boundary-hugging interpolated point falls back to bisection.
* **`HAGER_ZHANG`** — the OptimKit/TeneT.c line search (Algorithm 851:
  CG_DESCENT, Hager & Zhang, ACM TOMS 32 (2006)) with exact + approximate
  Wolfe acceptance; the production choice of TeneT.jl, more forgiving in
  numerically flat regions (its epsilon-relaxed acceptance tolerates
  round-off-level function increases).

`ARMIJO` is a backtracking search whose contraction is polynomial-
interpolated and safeguarded into ceres'
`[max_line_search_step_contraction, min_line_search_step_contraction]`
window.

Ceres semantics preserved where Xped relies on them: callbacks receive a
filled `IterationSummary` at iteration 0 and after every accepted iteration;
termination honors `max_num_iterations` / `function_tolerance` /
`gradient_tolerance` (max-norm) / `parameter_tolerance`;
`GradientProblem::function()` exposes the functor (used by
`iPEPSSolverAD::getCTMSolver`); `Summary::FullReport()` is available. One
deliberate strengthening: `Evaluate()` is **always** called with a valid
gradient pointer (Xped's `Energy` functor requires this; the CTM+AD pass
computes the gradient anyway).

## Build

Nothing to do — this is the default when `XPED_USE_NLO=ON` with
`XPED_OPTIM_LIB=ceres`. To use the real ceres-solver instead:

```
cmake -DXPED_USE_EXTERNAL_CERES=ON ...     # plus vcpkg feature "external-ceres"
```

## Related: self-contained SU(2) coupling coefficients

In the same spirit, `include/Xped/Symmetry/WignerSelf.hpp` replaces the GNU
Scientific Library (Xped's only GSL use was `gsl_sf_coupling_{3j,6j,9j}`)
with self-contained Racah-formula Wigner symbols in extended precision
(validated against GSL to <= 1.3e-13 over ~2.5 million symbols, exhaustive up
to two_j = 12 and random up to two_j = 40). GSL can be re-enabled with
`-DXPED_USE_GSL_COEFFS=ON` (vcpkg feature "gsl-coeffs"); the WIGXJPF /
FASTWIGXJ backends of `SU2Wrappers.hpp` are untouched.

## Verification

```
g++ -std=c++20 -I include/selfnlo tests/selfnlo_lbfgs_tests.cpp -o lbfgs_tests && ./lbfgs_tests
g++ -std=c++20 -I include tests/selfnlo_wigner_tests.cpp -lgsl -lgslcblas -o wigner_tests && ./wigner_tests
```

plus `tests/selfad_xped_tensor_test.cpp`, which runs real Xped U(1) **and**
SU(2) block-sparse tensors (the SU(2) run exercises the self-contained Wigner
symbols through the full Clebsch-Gordan machinery) with AD gradients checked
against finite differences.
