all:
	g++ -std=c++14 -march=native -fopenmp -Ofast -fno-finite-math-only main.cpp -o 2D_Poisson -lblas -lgomp -lsuperlu
