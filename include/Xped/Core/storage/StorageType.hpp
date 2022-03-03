#ifndef XPED_STORAGE_TYPE_HPP_
#define XPED_STORAGE_TYPE_HPP_

#if defined XPED_USE_CONTIGUOUS_STORAGE
#    include "Xped/Core/storage/StorageType_contiguous.hpp"
#elif defined XPED_USE_VECOFMAT_STORAGE
#    include "Xped/Core/storage/StorageType_vecofmat.hpp"
#endif

#endif
