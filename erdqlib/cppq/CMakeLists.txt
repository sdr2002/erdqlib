cmake_minimum_required(VERSION 3.28)
project(cppq VERSION 0.1.0 LANGUAGES CXX)

# Create your executable like below 'hellocmake' app.
add_executable(hellocmake hellocmake.cpp)
set_property(TARGET hellocmake PROPERTY CXX_STANDARD 20)

add_executable(montecarlo0 montecarlo0.cpp)
set_property(TARGET montecarlo0 PROPERTY CXX_STANDARD 20)

## C++ in VS Code: Building your Code with CMake - https://www.youtube.com/watch?v=_BWU5mWqVA4&t=22s
#include(CTest)
#enable_testing()

#add_library(CppqLibBinomial SHARED
#        binomial/lecture1/BinModel01Sptr.cpp
#        binomial/lecture1/BinModel01Sptr.h
#        binomial/lecture1/Options03.cpp
#        binomial/lecture1/Options03.h
#        binomial/lecture1/ex1_8_singlescript.cpp
#        binomial/lecture1/ex1_8.cpp
#        binomial/lecture1/ex1_9.cpp
#
#        binomial/lecture2/BinModel02.cpp
#        binomial/lecture2/BinModel02.h
#        binomial/lecture2/Options04.cpp
#        binomial/lecture2/Options04.h
#        binomial/lecture2/Main08.cpp
#        binomial/lecture2/Options05.cpp
#        binomial/lecture2/Options05.h
#        binomial/lecture2/Main09.cpp
#        binomial/lecture2/Options06.cpp
#        binomial/lecture2/Options06.h
#        binomial/lecture2/Main10.cpp
#
#        binomial/lecture3/Options07.cpp
#        binomial/lecture3/Options07.h
#        binomial/lecture3/Main12.cpp
#        binomial/lecture3/Options08.cpp
#        binomial/lecture3/Options08.h
#        binomial/lecture3/BinLattice01.h
#        binomial/lecture3/BinLattice02.h
#        binomial/lecture3/Options09.cpp
#        binomial/lecture3/Options09.h
#        binomial/lecture3/Main14.cpp
#        binomial/lecture3/ex3_2.cpp
#)

add_library(CppqLibMonteCarlo SHARED
        montecarlo/BSModel01.cpp
        montecarlo/BSModel01.h
        montecarlo/PathDepOption01.cpp
        montecarlo/PathDepOption01.h
        montecarlo/Task.h
        montecarlo/OUModel.cpp
        montecarlo/OUModel.h
        montecarlo/ModelCommon.h
        montecarlo/Model.h
        montecarlo/ModelCommon.cpp
)

target_link_libraries(hellocmake CppqLibMonteCarlo)
target_link_libraries(montecarlo0 CppqLibMonteCarlo)