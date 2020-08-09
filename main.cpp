/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include<immintrin.h>
#include<xmmintrin.h>
using namespace std;

#define DOUBLE_SIZE 8
#define LINE_SIZE 20
typedef chrono::high_resolution_clock Clock;

const int m=1638400;	// DO NOT CHANGE!!
const int K=100000;	// DO NOT CHANGE!!

double logDataVSPrior(const double* dat_real,const double* dat_imag, const double* pri_real,const double* pri_imag, const double* ctf, const double* sigRcp, const int start,const int chunkSize, const double disturb,const int chunkId);

int main ( int argc, char *argv[] )
{   
    double *dat_real = (double*)_mm_malloc(DOUBLE_SIZE*m,64),*dat_imag=(double*)_mm_malloc(DOUBLE_SIZE*m,64);
    double *pri_real = (double*)_mm_malloc(DOUBLE_SIZE*m,64),*pri_imag=(double*)_mm_malloc(DOUBLE_SIZE*m,64);
    double *ctf = (double*)_mm_malloc(DOUBLE_SIZE*m,64);
    double *sigRcp = (double*)_mm_malloc(DOUBLE_SIZE*m,64);
    double *disturb = (double*)_mm_malloc(DOUBLE_SIZE*K,64);
    double *result=(double*)_mm_malloc(DOUBLE_SIZE*m,64);
    double dat0, dat1, pri0, pri1, ctf0, sigRcp0;


    /***************************
     * Read data from input.dat
     * *************************/
    ifstream fin;

    fin.open("input.dat");
    if(!fin.is_open())
    {
        cout << "Error opening file input.dat" << endl;
        exit(1);
    }
    int i=0;
    while( !fin.eof() ) 
    {
        fin >> dat0 >> dat1 >> pri0 >> pri1 >> ctf0 >> sigRcp0;
        dat_real[i] =dat0;
        dat_imag[i]=dat1;
        pri_real[i] =pri0;
        pri_imag[i]= pri1;
        ctf[i] = ctf0;
        sigRcp[i] = sigRcp0;
        i++;
        if(i == m) break;
    }
    fin.close();

    fin.open("K.dat");
    if(!fin.is_open())
    {
	cout << "Error opening file K.dat" << endl;
	exit(1);
    }
    i=0;
    while( !fin.eof() )
    {
	fin >> disturb[i];
	i++;
	if(i == K) break;
    }
    fin.close();

    /***************************
     * main computation is here
     * ************************/
    auto startTime = Clock::now(); 
//    FILE* fp=fopen("result.dat","w");
//    if(fp==NULL){
//        cout << "Error opening file for result" << endl;
//        exit(1);
//    }
    
    ofstream fout;
    fout.open("result.dat");
    if(!fout.is_open())
    {
         cout << "Error opening file for result" << endl;
         exit(1);
    }
    
    int thread_num=1,chunkSize=512,chunkNum=m/chunkSize,reduce_chunk=0;
    double* resBuffer=nullptr; 
        
    #pragma omp parallel
    { 
        #pragma omp single
        {
            thread_num=omp_get_num_threads();
            resBuffer=(double*)_mm_malloc(DOUBLE_SIZE*thread_num*K,64);
        }
        int tid=omp_get_thread_num();
        #pragma omp for schedule(guided)
        for(int chunkId=0;chunkId<chunkNum;chunkId++){
            int start=chunkSize*chunkId;
            if(chunkId==chunkNum-1)
                chunkSize=m-start;
            for (unsigned int t = 0; t < K; t++)
            {
                resBuffer[tid*K+t]+=logDataVSPrior(dat_real,dat_imag, pri_real,pri_imag, ctf, sigRcp,start,chunkSize, disturb[t],chunkId);
            }
        }

        #pragma omp for schedule(guided)
        for(int t=0;t<K;t++){
            #pragma ivdep
            #pragma unroll(4)
            for(int i=0;i<thread_num;i++){
                result[t]+=resBuffer[i*K+t];
            }
        }
    }

//    for(unsigned int t=0;t<K;t++){
//        fprintf(fp,"%d: %.6g\n",t+1,result[t]);
//    }
//    fclose(fp);

    for(unsigned int t = 0; t < K; t++)
    {
        fout << t+1 << ": " << result[t] << endl;
    }
    fout.close();
	
	auto endTime = Clock::now(); 

    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() << " microseconds" << endl;
    _mm_free(dat_real);
    _mm_free(dat_imag);
    _mm_free(pri_real);
    _mm_free(pri_imag);

    _mm_free(ctf);
    _mm_free(sigRcp);
    _mm_free(disturb);

    _mm_free(resBuffer);
    _mm_free(result);
    return EXIT_SUCCESS;
}

double logDataVSPrior(const double* dat_real,const double* dat_imag, const double* pri_real,const double* pri_imag, const double* ctf, const double* sigRcp,const int start, const int chunkSize, const double disturb,const int chunkId)
{   
    double result=0.0;
    #pragma ivdep
    #pragma unroll(8)
    for (int i=start;i<start+chunkSize;i++){
        double real=dat_real[i]-disturb*ctf[i]*pri_real[i];
        double imag=dat_imag[i]-disturb*ctf[i]*pri_imag[i];
        result+=(real*real+imag*imag)*sigRcp[i];
    }
    return result;
}
