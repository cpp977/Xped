// Self-contained Xped AD engine: stand-in for stan-math's eigen_plugins.h.
//
// This file is textually included *inside* the class body of Eigen::DenseBase
// (via Xped/Util/EigenPlugins.hpp -> EIGEN_DENSEBASE_PLUGIN), so it may only
// contain member declarations.
//
// stan-math's original plugin adds .val()/.adj()/.vi() coefficient views for
// Eigen matrices whose Scalar is stan::math::var or vari*.  Xped performs AD
// at the level of whole block-sparse tensors (one vari per tensor, see
// Xped/AD/vari_value.hpp), never with Eigen matrices of var scalars, so no
// members are required here.
