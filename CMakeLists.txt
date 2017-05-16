cmake_minimum_required(VERSION 3.7)
project(lab_6)

set(CMAKE_CXX_STANDARD 98)

set(SOURCE_FILES
        lab6-students/DataReader.cpp
        lab6-students/DataReader.h
        lab6-students/Evaluation.cpp
        lab6-students/Evaluation.h
        lab6-students/EvaluationMeasures.cpp
        lab6-students/EvaluationMeasures.h
        lab6-students/KNN.cpp
        lab6-students/KNN.h
        lab6-students/main.cpp
        lab6-students/Point.cpp
        lab6-students/Point.h
        ecoli.processed.txt
        "Lab6 - KNN.pdf")

add_executable(lab_6 ${SOURCE_FILES})