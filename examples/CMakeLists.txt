#########################
#       ipeps_ad        #
#########################
add_executable(ipeps_ad ipeps_ad.cpp)
target_link_libraries (ipeps_ad Xped)

#########################
#       fpeps_ad        #
#########################
add_executable(fpeps_ad fpeps_ad.cpp)
target_link_libraries (fpeps_ad Xped)

#########################
#    simple_update      #
#########################
add_executable(simple_update simple_update.cpp)
target_link_libraries (simple_update Xped)

#########################
#        ad_ramp        #
#########################
add_executable(ad_ramp ad_ramp.cpp)
target_link_libraries (ad_ramp Xped)

#########################
#        chi_ramp       #
#########################
add_executable(chi_ramp chi_ramp.cpp)
target_link_libraries (chi_ramp Xped)

#########################
#    calc_corrLength    #
#########################
# add_executable(calc_corrLength calc_corrLength.cpp)
# target_link_libraries (calc_corrLength Xped)
