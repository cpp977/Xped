function(set_project_options project_name)

  set(MSVC_OPTIONS_RELEASE
  )
  set(MSVC_OPTIONS_DEBUG
  )
  set(MSVC_OPTIONS_PROFILE
  )
  
  set(CLANG_OPTIONS_RELEASE
      -stdlib=libc++
      -std=c++17
      -O3
      -ferror-limit=5
      -fcolor-diagnostics
  )
  set(CLANG_OPTIONS_DEBUG
      -std=c++17
      -stdlib=libc++
      -O0
      -g
      -ferror-limit=5
      -fcolor-diagnostics
  )
  set(CLANG_OPTIONS_PROFILE
      -std=c++17
      -stdlib=libc++
      -O2
      -pg
      -ferror-limit=5
      -fcolor-diagnostics
  )
  
  set(GCC_OPTIONS_RELEASE
      -std=c++17
      -O3
      -fmax-errors=5
      -fdiagnostics-color=always
  )
  set(GCC_OPTIONS_DEBUG
      -std=c++17
      -O0
      -g
      -fmax-errors=5
      -fdiagnostics-color=always
  )
  set(GCC_OPTIONS_PROFILE
      -std=c++17
      -O2
      -pg
      -fmax-errors=5
      -fdiagnostics-color=always
  )

  if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
   if(MSVC)
     set(PROJECT_OPTIONS ${MSVC_OPTIONS_RELEASE})
   elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
     set(PROJECT_OPTIONS ${CLANG_OPTIONS_RELEASE})
   elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
     set(PROJECT_OPTIONS ${GCC_OPTIONS_RELEASE})
   else()
     message(AUTHOR_WARNING "No compiler options set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
   endif()
  elseif (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
   if(MSVC)
     set(PROJECT_OPTIONS ${MSVC_OPTIONS_DEBUG})
   elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
     set(PROJECT_OPTIONS ${CLANG_OPTIONS_DEBUG})
   elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
     set(PROJECT_OPTIONS ${GCC_OPTIONS_DEBUG})
   else()
     message(AUTHOR_WARNING "No compiler options set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
   endif()
  elseif (${CMAKE_BUILD_TYPE} STREQUAL "Profile")
   if(MSVC)
     set(PROJECT_OPTIONS ${MSVC_OPTIONS_PROFILE})
   elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
     set(PROJECT_OPTIONS ${CLANG_OPTIONS_PROFILE})
   elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
     set(PROJECT_OPTIONS ${GCC_OPTIONS_PROFILE})
   else()
     message(AUTHOR_WARNING "No compiler options set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
   endif()
  endif()

target_compile_options(${project_name} INTERFACE ${PROJECT_OPTIONS})

if(ENABLE_LRU_CACHE)
target_compile_definitions(${project_name} INTERFACE CACHE_PERMUTE_OUTPUT=1)
endif()

endfunction()
