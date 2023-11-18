# vim:ts=2:et
#=============================================================================#
#                        "CMakeLists.txt" (Top-Level):                        #
#=============================================================================#
#=============================================================================#
# CMake Settings:                                                             #
#=============================================================================#
# Export flags for convenience when using Vim's YCM plugin:
#
CMAKE_MINIMUM_REQUIRED(VERSION 3.23)
CMAKE_POLICY(          VERSION 3.23)

IF (NOT UNIX)
  MESSAGE(FATAL_ERROR "Non-UNIX platforms are not supported")
ENDIF()

#=============================================================================#
# Project Settings:                                                           #
#=============================================================================#
IF (NOT CMAKE_CXX_COMPILER                 OR NOT ENV_PREFIX       OR
    NOT TOOL_CHAIN OR NOT CMAKE_BUILD_TYPE OR NOT PROJ_NAME        OR
    NOT LIB_DIR    OR NOT BIN_DIR          OR NOT DEFINED UNCHECKED_MODE)
  MESSAGE(FATAL_ERROR "Required Definition(s) Missing")
ENDIF()

# NB: In this case, PROJECT_SOURCE_DIR = CMAKE_SOURCE_DIR
#              and  PROJECT_BINARY_DIR = CMAKE_BINARY_DIR ,
# so we use the "CMAKE_"-prefixed vars:
#
PROJECT(${PROJ_NAME} LANGUAGES CXX)

SET (CMAKE_EXPORT_COMPILE_COMMANDS ON)
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

# Generate identity string for this build:
EXECUTE_PROCESS(
  COMMAND bash -c "cd ${CMAKE_SOURCE_DIR} && git rev-parse HEAD"
  OUTPUT_VARIABLE COMMIT)
STRING(STRIP ${COMMIT} COMMIT_NAME)

EXECUTE_PROCESS(
  COMMAND bash -c "cd ${CMAKE_SOURCE_DIR} && git rev-parse --abbrev-ref HEAD"
  OUTPUT_VARIABLE BRANCH)
STRING(STRIP ${BRANCH} BRANCH_NAME)

EXECUTE_PROCESS(
  COMMAND bash -c "cd ${CMAKE_SOURCE_DIR} && git config --get remote.origin.url"
  OUTPUT_VARIABLE REMOTE)
STRING(STRIP ${REMOTE} REMOTE_NAME)

STRING(TIMESTAMP DATE "%Y-%m-%d %H-%M-%S" UTC)

ADD_COMPILE_DEFINITIONS(REMOTE_NAME="${REMOTE_NAME}"
                        BRANCH_NAME="${BRANCH_NAME}"
                        COMPILE_DATE="${DATE}"
                        COMMIT_NAME="${COMMIT_NAME}")
MESSAGE(STATUS
  "Remote\t${REMOTE_NAME}\nBranch\t${BRANCH_NAME}\n\
Commit\t${COMMIT_NAME}\nTime\t${DATE}\n")

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${LIB_DIR}")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIB_DIR}")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}")

#-----------------------------------------------------------------------------#
# Other options:                                                              #
#-----------------------------------------------------------------------------#
# Crypto-Only mode (ignoring connectors to "Fiat" exchanges):
#IF (NOT DEFINED CRYPTO_ONLY OR "${CRYPTO_ONLY}" STREQUAL "OFF")
#  MESSAGE(STATUS "Crypto-Only : OFF")
#  SET(CRYPTO_ONLY 0)
#ELSE()
#  MESSAGE(STATUS "Crypto-Only : ON")
#  SET(CRYPTO_ONLY 1)
#ENDIF()

#=============================================================================#
# Check for Dependencies:                                                     #
#=============================================================================#
SET(EXT_LIBS)
#-----------------------------------------------------------------------------#
# Our Own Dependencies: MAQUETTE-Core, LibStatArb and MQTCommon:              #
#-----------------------------------------------------------------------------#
# All those projects must be siblings of "MAQUETTE-Strats":
# "MAQUETTE-Core":
#
#SET(MQT_CORE_TOP_DIR   "${CMAKE_SOURCE_DIR}/../MAQUETTE-Core")
#IF (NOT IS_DIRECTORY   "${MQT_CORE_TOP_DIR}")
#  MESSAGE(FATAL_ERROR  "${MQT_CORE_TOP_DIR} not found")
#ENDIF()
#SET(MQT_CORE_LIBDIR
#    "${MQT_CORE_TOP_DIR}/__BUILD__/${TOOL_CHAIN}-${BUILD_MODE}/lib")
#IF (NOT IS_DIRECTORY "${MQT_CORE_LIBDIR}")
#  MESSAGE(FATAL_ERROR
#          "MAQUETTE-Core Build not found in: ${MQT_CORE_LIBDIR}")
#ENDIF()
#LIST(APPEND EXT_LIBS "MQTCore3")
#
## "LibStatArb":
##
#SET(LSA_TOP_DIR        "${CMAKE_SOURCE_DIR}/../LibStatArb")
#IF (NOT IS_DIRECTORY   "${LSA_TOP_DIR}")
#  MESSAGE(FATAL_ERROR  "${LSA_TOP_DIR} not found")
#ENDIF()
#SET(LSA_LIBDIR
#    "${LSA_TOP_DIR}/__BUILD__/${TOOL_CHAIN}-${BUILD_MODE}/lib")
#IF (NOT IS_DIRECTORY "${LSA_LIBDIR}")
#  MESSAGE(FATAL_ERROR
#          "LibStatArb Build not found in: ${MQT_COMMON_LIBDIR}")
#ENDIF()
#LIST(APPEND EXT_LIBS "StatArb")
#
## "MQTCommon" lib must be the last in the sequence:
##
#SET(MQT_COMMON_TOP_DIR "${CMAKE_SOURCE_DIR}/../MQTCommon")
#IF (NOT IS_DIRECTORY   "${MQT_COMMON_TOP_DIR}")
#  MESSAGE(FATAL_ERROR  "${MQT_COMMON_TOP_DIR} not found")
#ENDIF()
#SET(MQT_COMMON_LIBDIR
#    "${MQT_COMMON_TOP_DIR}/__BUILD__/${TOOL_CHAIN}-${BUILD_MODE}/lib")
#IF (NOT IS_DIRECTORY "${MQT_COMMON_LIBDIR}")
#  MESSAGE(FATAL_ERROR
#          "MQTCommon Build not found in: ${MQT_COMMON_LIBDIR}")
#ENDIF()
#LIST(APPEND EXT_LIBS "MQTCommon")
#
## Addition of questdb lib which was compiled into separate lib from a 
## Rust package
#LIST(APPEND EXT_LIBS "${MQT_COMMON_LIBDIR}/libquestdb_client.a")

#-----------------------------------------------------------------------------#
# System Dependencies: Boost, GNUTLS, NGHTTP2, WT, etc:                       #
#-----------------------------------------------------------------------------#
# XXX: UNFORTUNATELY, due to static linking of MAQUETTE-Core and using header-
# only libraries, we currently have to use the same long list of system depen-
# dencies here as in MAQUETTE-Core:
#
# Extend the PkgConfig search path (for *.pc files) -- useful for those modules
# which can be found via PkgConfig:
#SET(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${ENV_PREFIX}/lib/pkgconfig")
#
## Boost (no specific version checks);  in the MAQUETTE Core, it is used as a
## Headers-Only library:
#FIND_PACKAGE(Boost  REQUIRED)
#
## GNUTLS:
#FIND_PACKAGE(GnuTLS REQUIRED)
#LIST(APPEND EXT_LIBS "gnutls")
#
## NGHTTP2  for HTTP2 header compression:
#FIND_PACKAGE(NGHTTP2 REQUIRED)
#LIST(APPEND EXT_LIBS "nghttp2")
#
## Wt: For monitoring tools:
#SET(WITH_WT 0)
#FIND_PACKAGE(Wt)
#IF (Wt_FOUND)
#  SET (WITH_WT 1)
#  LIST(APPEND EXT_LIBS "wt" "wthttp")
#ENDIF()

#=============================================================================#
# Compiler Settings:                                                          #
#=============================================================================#
MESSAGE(STATUS "Configuring for the ${TOOL_CHAIN} ToolChain...")

#-----------------------------------------------------------------------------#
IF("${TOOL_CHAIN}" STREQUAL "GCC")
#-----------------------------------------------------------------------------#
  # Optimisation and debugging flags:
  IF ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    # NB: Do NOT use:
    # "-ffast-math" option -- it will result in incorrect treatment of "NaN"s!
    # -flto"               -- it intereacts in a strange way with header-only
    #                         code!
    SET(CXXFLAGS "-O3 -mavx2 -DNDEBUG -funroll-loops -fomit-frame-pointer")
  ELSEIF ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    SET(CXXFLAGS "-O2 -g -funroll-loops")
  ELSEIF ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    SET(CXXFLAGS "-O0 -g")
  ELSE()
    MESSAGE(WARNING "UnSupported BuildType: ${CMAKE_BUILD_TYPE}")
  ENDIF()

  # Other flags:
  SET(CXXFLAGS "${CXXFLAGS} -march=native -mtune=native -std=gnu++23 \
                -Wall                     -Wextra            \
                -Wctor-dtor-privacy       -Wnon-virtual-dtor \
                -Woverloaded-virtual      -Wcast-qual        \
                -Winit-self               -Wwrite-strings    \
                -Wmissing-braces          -Wdisabled-optimization  \
                -Wconversion              -Wno-stringop-truncation \
                -Wno-class-memaccess      -Wno-double-promotion    \
                -Wno-c++20-compat    -fopenmp -lpthread" )

#-----------------------------------------------------------------------------#
ELSEIF("${TOOL_CHAIN}" STREQUAL "CLang")
#-----------------------------------------------------------------------------#
  # Optimisation and debugging flags:
  IF ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    SET(CXXFLAGS "-O3 -mavx2 -DNDEBUG -funroll-loops -fomit-frame-pointer")
  ELSEIF ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    SET(CXXFLAGS "-O2 -g -funroll-loops")
  ELSEIF ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"  )
    SET(CXXFLAGS "-O0 -g")
  ELSE()
    MESSAGE(WARNING "UnSupported BuildType: ${CMAKE_BUILD_TYPE}")
  ENDIF()

  # Other flags (common C++/C, and C++-only):
  SET(CXXFLAGS "${CXXFLAGS}  -march=native -mtune=native                 \
                -Weverything -std=gnu++23  -Wno-unsafe-buffer-usage      \
                -Wno-c++98-compat          -Wno-c++98-compat-pedantic    \
                -Wno-weak-vtables          -Wno-double-promotion         \
                -Wno-weak-template-vtables -Wno-covered-switch-default   \
                -Wno-switch-enum           -Wno-exit-time-destructors    \
                -Wno-global-constructors   -Wno-reserved-id-macro        \
                -Wno-padded                -Wno-unused-macros            \
                -Wno-gnu-zero-variadic-macro-arguments \
                -Wno-disabled-macro-expansion          \
                -Wno-float-equal           -Wno-missing-prototypes       \
                -Wno-vla                   -Wno-vla-extension            \
                -Wno-c99-extensions        -Wno-zero-length-array        \
                -Wno-packed                -Wno-undefined-func-template  \
                -Wno-gnu-statement-expression")

#-----------------------------------------------------------------------------#
ELSEIF("${TOOL_CHAIN}" STREQUAL "NVHPC")
#-----------------------------------------------------------------------------#
  # Optimisation and debugging flags:
  IF ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    SET(CXXFLAGS "-O3 -mavx2 -DNDEBUG")
  ELSEIF ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    SET(CXXFLAGS "-O2 -g")
  ELSEIF ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"  )
    SET(CXXFLAGS "-O0 -g")
  ELSE()
    MESSAGE(WARNING "UnSupported BuildType: ${CMAKE_BUILD_TYPE}")
  ENDIF()

  # Other flags (common C++/C, and C++-only):
  SET(CXXFLAGS  "${CXXFLAGS} -march=native -mtune=native    \
                -Wno-switch-enum           -Wno-float-equal \
                -Wno-missing-prototypes    -Wno-vla         -std=c++23")
#-----------------------------------------------------------------------------#
ELSE()
  MESSAGE(FATAL_ERROR  "UnSupported ToolChain: ${TOOL_CHAIN}")
ENDIF()

#=============================================================================#
# Compiler-Independent Compilation Env:                                       #
#=============================================================================#
# Reset the standard flags for the selected mode, to prevent a mix-up:
SET(CMAKE_CXX_FLAGS_RELEASE        "")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")
SET(CMAKE_CXX_FLAGS_DEBUG          "")

# Now set our own flags which already take the mode into account:
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXFLAGS}")

#-----------------------------------------------------------------------------#
# Special defs (mostly for Boost) and libs:                                   #
#-----------------------------------------------------------------------------#
ADD_DEFINITIONS(-DBOOST_SYSTEM_NO_DEPRECATED
                -UBOOST_CB_ENABLE_DEBUG
                -DBOOST_CB_DISABLE_DEBUG
                -DUNCHECKED_MODE=${UNCHECKED_MODE}
                -DWITH_WT=${WITH_WT}
                -DCRYPTO_ONLY=${CRYPTO_ONLY})

#-----------------------------------------------------------------------------#
# Common Configs:                                                             #
#-----------------------------------------------------------------------------#
# Use CCache if available:
OPTION(USE_CCACHE "Use ccache" OFF)
IF (USE_CCACHE)
  FIND_PROGRAM(CCACHE_FOUND ccache)
  IF (CCACHE_FOUND)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK    ccache)
  ENDIF()
ENDIF()

# During build, link executables with build rpath, not install rpath; they are
# then automatically re-linked on install. This is the default behaviour, any-
# way:
SET(CMAKE_SKIP_BUILD_RPATH         FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Src
  #                    3rdParty
  #                    ${MQT_COMMON_TOP_DIR}/Include
  #                    ${MQT_COMMON_TOP_DIR}/3rdParty
  #                  ${MQT_CORE_TOP_DIR}/Src
                    ${ENV_PREFIX}/Include
  #                  ${NLOPT_INCLUDE_DIRS})
  )
LINK_DIRECTORIES   (${LIB_DIR}
  #${MQT_COMMON_LIBDIR}
  #                 ${LSA_LIBDIR}
  #                 ${MQT_CORE_LIBDIR}
  #                 ${ENV_PREFIX}/lib
  )



# Ignore warnings in 3rd-party headers (incl UTXX!):
# XXX: Using ADD_DEFINITIONS to that end is not a very good idea, but we do it
# anyway, in order not to change CMAKE{C|CXX}_FLAGS anymore:
#
#ADD_DEFINITIONS    (-isystem "${MQT_COMMON_TOP_DIR}/3rdParty"
#                    -isystem "${ENV_PREFIX}/Include")

#=============================================================================#
# Strats to be Built:                                                         #
#=============================================================================#
#SET(MQT_STRATS
## TriArbC0
## PMoment
## MM-Single   # Temporarily de-configured
## CryptoDataRecorder
#)
#
## Generic Settings (for all Strategies):
##
#FOREACH(Strat ${MQT_STRATS})
#  ADD_EXECUTABLE(       ${Strat} ${Strat}/${Strat}.cpp)
#  SET_TARGET_PROPERTIES(${Strat} PROPERTIES OUTPUT_NAME ${Strat})
#  TARGET_LINK_LIBRARIES(${Strat} ${EXT_LIBS})
#
#  # Build the WebServer controlling that Strategy (if required) and link it
#  # with Wt libraries (if exist!):
#  IF (WITH_WT)
#    SET(WTStrat    "WT-${Strat}")
#    SET(WTStratSrc "${Strat}/${WTStrat}.cpp")
#
#    # NB: Need a full path here, or "WTStratSrc" would not be found!
#    IF (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${WTStratSrc}")
#      ADD_EXECUTABLE(       ${WTStrat} ${WTStratSrc})
#      SET_TARGET_PROPERTIES(${WTStrat} PROPERTIES OUTPUT_NAME ${WTStrat})
#      TARGET_LINK_LIBRARIES(${WTStrat} ${EXT_LIBS})
#    ENDIF()
#  ENDIF()
#ENDFOREACH(Strat)

ADD_EXECUTABLE(test main.cpp)

# Strategy-Specific Settings:
# Temporarily de-configured:
# TARGET_LINK_LIBRARIES(MM-Single "curl")
