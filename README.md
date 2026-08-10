# catboostr

catboostr is an R package for CatBoost, the gradient-boosting-on-decision-trees
library with built-in categorical-feature support. It is a fork of upstream
CatBoost's R package, pinned to CatBoost v1.2.10
(`b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`), and its goal is a strict
superset of upstream's `catboost.*` interface: every function and signature
that works in the upstream package keeps working here, with additional
capability layered on top. R1, the release this README describes, covers
everything CPU-based; GPU support is deferred to a later phase because the
project has no CUDA hardware to verify it against.

## Install

Install from a git checkout:

```bash
git clone https://github.com/davdittrich/catboostr.git
cd catboostr
tools/vendor/acquire.sh          # fetches the pinned CatBoost v1.2.10 source
R CMD INSTALL --preclean .
```

`tools/vendor/acquire.sh` clones upstream CatBoost at the pinned tag into
`vendor/catboost` and verifies its commit SHA before doing anything else; it
exits non-zero if the SHA doesn't match, rather than installing an unverified
tree. `R CMD INSTALL` builds `libcatboostr` from that vendored source via
CMake, so you also need CMake (>= 3.15), a C++20-capable compiler
(clang/clang++), Python3, Cython, Perl, and GNU Make or Ninja on your `PATH`
-- see `SystemRequirements` in `DESCRIPTION`. Installing performs no network
access beyond the `acquire.sh` step; everything after that builds from the
local checkout.

## Platform support

Linux and macOS are both verified on every push via GitHub Actions CI, using
GitHub-hosted runners -- no local Linux or macOS hardware is required to
reproduce these results yourself. Windows is deferred: CatBoost's Windows
build path targets MSVC/clang-cl, while R on Windows links against
Rtools' mingw-w64 toolchain, and the two produce C++ ABIs that don't link
against each other.

## GPU

`task_type = "GPU"` is real upstream surface and is preserved, but this
build is CPU-only: passing it fails with an explicit error identifying the
missing CUDA device, rather than silently falling back to CPU. GPU parity
is deferred until CUDA hardware is available to verify it against
differentially, tracked as a later project phase.

## Parity and documentation

[`docs/PARITY.md`](docs/PARITY.md) is a machine-generated matrix comparing
every CatBoost R, Python, and CLI capability, regenerated from
`tests/fixtures/parity/matrix.dispositioned.json`. It lists what's verified
equivalent across the three interfaces, what's out of scope, and what's a
genuine, tracked gap.

## License

Apache License 2.0, inherited from upstream CatBoost -- see the `License`
field in `DESCRIPTION`.
