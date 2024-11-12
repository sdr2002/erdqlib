cmake_minimum_required(VERSION 3.29)
project(city_ir_fx)

set(CMAKE_CXX_STANDARD 23)

add_executable(city_ir_fx
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

        montecarlo/BSModel01.cpp
        montecarlo/BSModel01.h
        montecarlo/PathDepOption01.cpp
        montecarlo/PathDepOption01.h
        montecarlo/Main.cpp
        montecarlo/OUModel.cpp
        montecarlo/OUModel.h
        montecarlo/ModelCommon.h
        montecarlo/Model.h
        montecarlo/ModelCommon.cpp
)
