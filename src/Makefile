# No-op companion to ../configure.
#
# configure already builds libcatboostr via the fork's own CMake target
# (src/CMakeLists.txt) and installs the result into inst/libs/libcatboostr.so
# BEFORE R CMD INSTALL reaches this directory. R's install machinery copies
# inst/* verbatim into the installed package (-> <pkg>/libs/libcatboostr.so,
# matching NAMESPACE's useDynLib(libcatboostr)), so nothing needs compiling
# here. This file's only job is to exist: R CMD INSTALL runs `make` in src/
# when a Makefile is present instead of falling back to its own R-CMD-SHLIB
# rule, which would otherwise try (and fail) to compile src/*.cpp itself
# with none of the include/link flags the CMake build supplies.
all:
clean:
