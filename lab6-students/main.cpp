#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "KNN.h"
#include "DataReader.h"
#include "EvaluationMeasures.h"
#include "Evaluation.h"

using std::string;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;

void func_q1(vector<Point> &allData) // question 1: K=1 classifier, train entire set, predict entire set.
{
    cout<<"question 1:"<<endl;
    KNN knn(1);
    knn.train(allData);
    for(int i=0;i<allData.size();i++)
        knn.predict(allData[i]);
    cout<<"accuracy="<<EvaluationMeasures::accuracy(allData)<<endl;
}

void func_q2(vector<Point> &allData) // question 2: 1<=K<=30 classifier, LOOCV.
{
    cout<<"question 2:"<<endl;
    for(int k=1;k<31;k++)
    {
        KNN classifier(k);
        Evaluation eval(classifier);
        cout<<"K="<<k<<", accuracy="<<eval.crossValidation(allData,allData.size())<<endl;
    }
}

void func_q3(vector<Point> &allData,string reqCV)
{
    unsigned cv(0);
    KNN classifier(9); // selected best classifier - k=9 - in Q2
    Evaluation eval(classifier);
    if(reqCV=="twofold")
        cv=2;
    else if(reqCV=="tenfold")
        cv=10;
    else if(reqCV=="loocv")
        cv=allData.size();
    else return;
    eval.crossValidation(allData,cv);
}

int main(int argc, char *argv[])
{
    if (argc<2)
    {
        cerr << "You are missing the input file name" << endl;
        return 1;
    }
    string fileName(argv[1]);
    DataReader dr;
    vector<Point> allData;
    dr.read(fileName, allData);
    cout << allData.size() << endl;
//    func_q1(allData); // question 1
    func_q2(allData); // question 2
//    func_q3(allData,"twofold"); // question 3 - 2-fold CV
//    func_q3(allData,"tenfold"); // question 3 - 10-fold CV
//    func_q3(allData,"loocv"); // question 3 - LOO-CV
    return 0;
}