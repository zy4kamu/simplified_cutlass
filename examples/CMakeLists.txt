simplified_cublass_add_executable(example basic_gemm.cu)

install(
  TARGETS example
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
