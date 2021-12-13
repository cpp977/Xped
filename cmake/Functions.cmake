function(create_plain_name plainlib vectorlib matrixlib tensorlib)
  if((${vectorlib} STREQUAL ${matrixlib}) AND (${vectorlib} STREQUAL ${tensorlib}))
    set(${plainlib} ${vectorlib} PARENT_SCOPE)
  elseif(${vectorlib} STREQUAL "Eigen" AND ${matrixlib} STREQUAL "Eigen" AND ${tensorlib} STREQUAL "Array")
    set(${plainlib} "Eigen_Array" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "You provided an unsupported combination of linear algebra libraries. (${vectorlib}, ${matrixlib}, ${tensorlib})")
  endif()
endfunction()