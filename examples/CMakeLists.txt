#########################
#        peps_ad        #
#########################
add_executable(peps_ad peps_ad.cpp)
target_link_libraries (peps_ad Xped)

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
