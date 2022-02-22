#ifndef XPED_STORAGE_TYPE_HPP_
#define XPED_STORAGE_TYPE_HPP_

#if defined XPED_USE_CONTIGOUS_STORAGE
#    include "Xped/Core/storage/StorageType_contigous.hpp"
#elif defined XPED_USE_VECOFMAT_STORAGE
#    pragma message("VecOfMat storage")
#    include "Xped/Core/storage/StorageType_vecofmat.hpp"
#endif

#endif
