function(set_project_options project_name)

  set(MSVC_OPTIONS_RELEASE
  )
  set(MSVC_OPTIONS_DEBUG
  )
  set(MSVC_OPTIONS_PROFILE
  )

  set(MSVC_LOPTIONS_RELEASE
  )
  set(MSVC_LOPTIONS_DEBUG
  )
  set(MSVC_LOPTIONS_PROFILE
  )
  if(USE_LIBCXX)
   set(USED_LIBCXX libc++)
  else()
   set (USED_LIBCXX libstdc++)
  endif()
  
  set(CLANG_OPTIONS_RELEASE
      -stdlib=${USED_LIBCXX}
      -std=c++17
      -O3
      -DNDEBUG
      -march=native
      -m64
      -ferror-limit=5
      -fcolor-diagnostics
  )
  set(CLANG_OPTIONS_DEBUG
      -std=c++17
      -stdlib=${USED_LIBCXX}
      -O0
      -g
      -march=native
      -m64
      -ferror-limit=5
      -fcolor-diagnostics
  )
  set(CLANG_OPTIONS_PROFILE
      -std=c++17
      -stdlib=${USED_LIBCXX}
      -O2
      -pg
      -march=native
      -m64
      -ferror-limit=5
      -fcolor-diagnostics
  )

  set(CLANG_LOPTIONS_RELEASE
      -stdlib=${USED_LIBCXX}
      -O3
      -DNDEBUG
  )
  set(CLANG_LOPTIONS_DEBUG
      -stdlib=${USED_LIBCXX}
      -O0
      -g
  )
  set(CLANG_LOPTIONS_PROFILE
      -stdlib=${USED_LIBCXX}
      -O2
      -pg
  )
  
  set(GCC_OPTIONS_RELEASE
      -std=c++17
      -O3
      -DNDEBUG
      -march=native
      -m64
      -fmax-errors=5
      -fdiagnostics-color=always
  )
  set(GCC_OPTIONS_DEBUG
      -std=c++17
      -O0
      -g
      -march=native
      -m64
      -fmax-errors=5
      -fdiagnostics-color=always
  )
  set(GCC_OPTIONS_PROFILE
      -std=c++17
      -O2
      -pg
      -march=native
      -m64
      -fmax-errors=5
      -fdiagnostics-color=always
  )

  set(GCC_LOPTIONS_RELEASE
      -O3
      -DNDEBUG
  )
  set(GCC_LOPTIONS_DEBUG
      -O0
      -g
  )
  set(GCC_LOPTIONS_PROFILE
      -O2
      -pg
  )
  
  if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
   if(MSVC)
     set(PROJECT_OPTIONS ${MSVC_OPTIONS_RELEASE})
     set(PROJECT_LOPTIONS ${MSVC_LOPTIONS_RELEASE})
   elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
     set(PROJECT_OPTIONS ${CLANG_OPTIONS_RELEASE})
     set(PROJECT_LOPTIONS ${CLANG_LOPTIONS_RELEASE})
   elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
     set(PROJECT_OPTIONS ${GCC_OPTIONS_RELEASE})
     set(PROJECT_LOPTIONS ${GCC_LOPTIONS_RELEASE})
   else()
     message(AUTHOR_WARNING "No compiler options set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
   endif()
  elseif (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
   if(MSVC)
     set(PROJECT_OPTIONS ${MSVC_OPTIONS_DEBUG})
     set(PROJECT_LOPTIONS ${MSVC_LOPTIONS_DEBUG})
   elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
     set(PROJECT_OPTIONS ${CLANG_OPTIONS_DEBUG})
     set(PROJECT_LOPTIONS ${CLANG_LOPTIONS_DEBUG})
   elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
     set(PROJECT_OPTIONS ${GCC_OPTIONS_DEBUG})
     set(PROJECT_LOPTIONS ${GCC_LOPTIONS_DEBUG})
   else()
     message(AUTHOR_WARNING "No compiler options set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
   endif()
  elseif (${CMAKE_BUILD_TYPE} STREQUAL "Profile")
   if(MSVC)
     set(PROJECT_OPTIONS ${MSVC_OPTIONS_PROFILE})
     set(PROJECT_LOPTIONS ${MSVC_LOPTIONS_PROFILE})
   elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
     set(PROJECT_OPTIONS ${CLANG_OPTIONS_PROFILE})
     set(PROJECT_LOPTIONS ${CLANG_LOPTIONS_PROFILE})
   elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
     set(PROJECT_OPTIONS ${GCC_OPTIONS_PROFILE})
     set(PROJECT_LOPTIONS ${GCC_LOPTIONS_PROFILE})
   else()
     message(AUTHOR_WARNING "No compiler options set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
   endif()
  endif()

target_compile_options(${project_name} INTERFACE ${PROJECT_OPTIONS})

target_link_options(${project_name} INTERFACE ${PROJECT_LOPTIONS})

if(ENABLE_LRU_CACHE)
target_compile_definitions(${project_name} INTERFACE CACHE_PERMUTE_OUTPUT=1)
endif()

endfunction()
