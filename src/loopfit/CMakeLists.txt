add_cython_target(_wrap CXX)
add_cython_target(_utils CXX)

add_library(_wrap MODULE ${_wrap} ceres_fit eqn_cubic)
add_library(_utils MODULE ${_utils})

python_extension_module(_wrap)
python_extension_module(_utils)

target_link_libraries(_wrap ${CERES_LIBRARIES})

install(TARGETS _wrap LIBRARY DESTINATION src/loopfit)
install(TARGETS _utils LIBRARY DESTINATION src/loopfit)