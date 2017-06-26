#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define IX(r,c,col) (c + r*col)

typedef struct {
    unsigned int row;
    unsigned int col;
    float *matrix;
} Matrix;

void setValueMatrix(Matrix *m, float value);
void readMatrix(Matrix *m, char *name);
void countLines(char *filename, int *r1, int *c1);
void randomizeMatrix(Matrix *m);
void addScalar(Matrix *m, float scalar);
void multiplyScalar(Matrix *m, float scalar);
void getSubMatrix(Matrix *m, Matrix main_mat, int row1, int row2, int col1, int col2, int transpose);
void multiplyMatrix(Matrix *result, Matrix m1, Matrix m2);
void addSubtractMatrix(Matrix *result, Matrix m1, Matrix m2, int sub);
void computeY(Matrix *return_mat, Matrix m);
void shuffleArray(int *array, size_t n);
void computeDeltaOut(Matrix *m, Matrix err, Matrix out);
void computeDeltaHidden(Matrix *m, Matrix y, Matrix w, Matrix d);
void transposeMatrix(Matrix *m, Matrix main);
void updateWeights(Matrix *weight, Matrix *weight_tmp, Matrix delta, Matrix y, Matrix *y_t, float eta);
void updateBias(Matrix *bias, Matrix delta, float eta);
void outputScores(FILE *fout, int q, float precision[8], float recall[8], float f1score[8], double f1score_mean, double mse);
void printScores(int confusion[8][8], int tp[8], int fp[8], int fn[8], float precision[8], float recall[8], float f1score[8]);
Matrix createNewMatrix(unsigned int nrow, unsigned int ncol);
Matrix openFile(char *filename);