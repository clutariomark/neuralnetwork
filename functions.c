#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "functions.h"

#define IX(r,c,col) (c + r*col)

void setValueMatrix(Matrix *m, float value) {

    int nrow = m->row;
    int ncol = m->col;
    for (int r = 0; r < nrow; r++) {
        for (int c = 0; c < ncol; c++) {
            m->matrix[IX(r,c,ncol)] = value;
        }
    }
}

void readMatrix(Matrix *m, char *name) {
    int col, row;
    int ncol = m->col;
    int nrow = m->row;
    if (ncol <= 4) col = ncol;
    else col = 4;
    if (nrow <= 4) row = nrow;
    else row = 4;
    
    printf("matrix: %s | ", name);
    printf("size: %lu | ", nrow*ncol*sizeof(float));
    printf("row: %d/%d | col: %d/%d\n", row, nrow, col, ncol);

    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            printf("%lf ", m->matrix[IX(r,c,ncol)]);
        }
        printf("\n");
    }
    printf("\n");
}

void countLines(char *filename, int *r1, int *c1) {
    FILE *fp;
    int row = 0;
    int col = 0;
    char c;
    
    fp = fopen(filename, "r");
    for (c = getc(fp); c != EOF; c = getc(fp)) {
        if (c == '\n') {
            row++;
        }
        if (c == ',') {
            col++;
        }
    }
    fclose(fp);
    
    *r1 = row;
    *c1 = col/row + 1;
}

void addScalar(Matrix *m, float scalar) {
    for (int r = 0; r < m->row; r++) {
        for (int c = 0; c < m->col; c++) {
            m->matrix[IX(r,c,m->col)] = m->matrix[IX(r,c,m->col)] + scalar;
        }
    }
}

void multiplyScalar(Matrix *m, float scalar) {
    for (int r = 0; r < m->row; r++) {
        for (int c = 0; c < m->col; c++) {
            m->matrix[IX(r,c,m->col)] = m->matrix[IX(r,c,m->col)]*scalar;
        }
    }
}


void randomizeMatrix(Matrix *m) {
    srand(time(NULL));
    for (int r = 0; r < m->row; r++) {
        for (int c = 0; c < m->col; c++) {
            m->matrix[IX(r,c,m->col)] = (float)rand()/(float)((unsigned)RAND_MAX + 1);
        }
    }
}

void getSubMatrix(Matrix *m, Matrix main_mat, int row1, int row2, int col1, int col2, int transpose) {
    int newrow, newcol;
    if (transpose == 1) {
        newrow = col2 - col1;
        newcol = row2 - row1;
    } else {
        newrow = row2 - row1;
        newcol = col2 - col1;
    }
    
    for (int r = row1; r < row2; r++) {
        for (int c = col1; c < col2; c++) {
            int new_c, new_r;
            if (transpose == 1) {
                new_r = c - col1;
                new_c = r - row1;
            } else {
                new_r = r - row1;
                new_c = c - col1;
            }
            
            m->matrix[IX(new_r,new_c,newcol)] = main_mat.matrix[IX(r,c,main_mat.col)];
        }
    }
}

void multiplyMatrix(Matrix *result, Matrix m1, Matrix m2) {
    if (m1.col != m2.row) {
        printf("ERROR ON [multiplyMatrix]:\n");
        printf("E >> Matrix 1 column not equal to Matrix 2 row.\n");
        printf("E >> This will return an empty matrix.\n\n");
    } else {
        for (int r1 = 0; r1 < m1.row; r1++) {
            for (int c2 = 0; c2 < m2.col; c2++) {
                float sum = 0;
                for (int c1 = 0; c1 < m1.col; c1++) {
                    sum = sum + m1.matrix[IX(r1,c1,m1.col)] * m2.matrix[IX(c1,c2,m2.col)];
                }
                result->matrix[IX(r1,c2,m2.col)] = sum;
            }
        }
    }
}

void addSubtractMatrix(Matrix *result, Matrix m1, Matrix m2, int sub) {
    int m1row = m1.row;
    int m2row = m2.row;
    int m1col = m1.col;
    int m2col = m2.col;
    if (m1row != m2row || m1col != m2col) {
        printf("ERROR ON [addSubtractMatrix]:\n");
        printf("E >> Matrix 1 and Matrix 2 does not have the same dimensions.\n");
        printf("E >> This will return an empty matrix.\n\n");
    } else {
        for (int r = 0; r < m1row; r++) {
            for (int c = 0; c < m1col; c++) {
                if (sub == 0)
                    result->matrix[IX(r,c,m1col)] = m1.matrix[IX(r,c,m1col)] + m2.matrix[IX(r,c,m2col)];
                else
                    result->matrix[IX(r,c,m1col)] = m1.matrix[IX(r,c,m1col)] - m2.matrix[IX(r,c,m2col)];
            }
        }
    }
}

void computeY(Matrix *return_mat, Matrix m) {
    for (int r = 0; r < m.row; r++) {
        for (int c = 0; c < m.col; c++) {
            return_mat->matrix[IX(r,c,m.col)] = 1.00/(1.00 + exp(-m.matrix[IX(r,c,m.col)]));
        }
    }
}

void shuffleArray(int *array, size_t n) {
    srand(time(NULL));
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX/(n-i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

void computeDeltaOut(Matrix *m, Matrix err, Matrix out) {
    if (err.row != out.row || err.col != out.col) {
        printf("ERROR ON [computeDelta]:\n");
        printf("E >> %d, %d, %d, %d\n", err.row, err.col, out.row, out.col);
        printf("E >> Matrix 1 and Matrix 2 does not have the same dimensions.\n");
        printf("E >> This will return an empty matrix.\n\n");
    } else {
        for (int r = 0; r < err.row; r++) {
            for (int c = 0; c < err.col; c++) {
                m->matrix[IX(r,c,m->col)] = err.matrix[IX(r,c,err.col)] * out.matrix[IX(r,c,out.col)] * (1.0 - out.matrix[IX(r,c,out.col)]);
            }
        }
    }
}

void computeDeltaHidden(Matrix *m, Matrix y, Matrix w, Matrix d) {    
    for (int r1 = 0; r1 < w.col; r1++) {
        for (int c2 = 0; c2 < d.col; c2++) {
            float sum = 0;
            for (int c1 = 0; c1 < w.row; c1++) {
                sum = sum + w.matrix[IX(r1,c1,w.row)] * d.matrix[IX(c1,c2,d.col)];
            }
            m->matrix[IX(r1,c2,d.col)] = sum;
        }
    }
    

    for (int r = 0; r < m->row; r++) {
        for (int c = 0; c < m->col; c++) {
            m->matrix[IX(r,c,m->col)] = y.matrix[IX(r,c,m->col)]*(1.0 - y.matrix[IX(r,c,m->col)]) * m->matrix[IX(r,c,m->col)];
        }

    }
}

void transposeMatrix(Matrix *m, Matrix main) {
    int row = main.col;
    int col = main.row;
    
    for (int r = 0; r < main.row; r++) {
        for (int c = 0; c < main.col; c++) {
            m->matrix[IX(c,r,col)] = main.matrix[IX(r,c,main.col)];
        }
    }
}

void updateWeights(Matrix *weight, Matrix *weight_tmp, Matrix delta, Matrix y, Matrix *y_t, float eta) {

    transposeMatrix(y_t, y);
    multiplyMatrix(weight_tmp, delta, *y_t);

    for (int r = 0; r < weight->row; r++) {
        for (int c = 0; c < weight->col; c++) {
            weight->matrix[IX(r,c,weight->col)] = weight->matrix[IX(r,c,weight->col)] + (eta * weight_tmp->matrix[IX(r,c,weight_tmp->col)]);
        }
    }
}

void updateBias(Matrix *bias, Matrix delta, float eta) {
    for (int r = 0; r < bias->row; r++) {
        for (int c = 0; c < bias->col; c++) {
            bias->matrix[IX(r,c,bias->col)] = bias->matrix[IX(r,c,bias->col)] + (eta * delta.matrix[IX(r,c,delta.col)]);
        }
    }
}

void outputScores(FILE *fout, int q, float precision[8], float recall[8], float f1score[8], double f1score_mean, double mse) {
    fprintf(fout, "%d,", q);
    for (int g = 0; g < 8; g++) {
        fprintf(fout, "%f,", precision[g]);
    }
    for (int g = 0; g < 8; g++) {
        fprintf(fout, "%f,", recall[g]);
    }
    for (int g = 0; g < 8; g++) {
        fprintf(fout, "%f,", f1score[g]);
    }
    fprintf(fout, "%f,", f1score_mean);
    fprintf(fout, "%f\n", mse);
}

void printScores(int confusion[8][8], int tp[8], int fp[8], int fn[8], float precision[8], float recall[8], float f1score[8]) {
    printf("confusion matrix\n");
    for (int b = 0; b < 8; b++) {
        for (int c = 0; c < 8; c++) {
            if (c == b)
                printf("[%04d] ", confusion[b][c]);
            else
                printf(" %04d  ", confusion[b][c]);
        }
        printf("\n");
    }
    printf("tp: ");
    for (int g = 0; g < 8; g++)
        printf("%d, ", tp[g]);
    printf("\n");
    printf("fp: ");
    for (int g = 0; g < 8; g++)
        printf("%d, ", fp[g]);
    printf("\n");
    printf("fn: ");
    for (int g = 0; g < 8; g++)
        printf("%d, ", fn[g]);
    printf("\n");
    printf("precision: ");
    for (int g = 0; g < 8; g++)
        printf("%.2f, ", precision[g]);
    printf("\n");
    printf("recall: ");
    for (int g = 0; g < 8; g++)
        printf("%.2f, ", recall[g]);
    printf("\n");
    printf("f1 score: ");
    
    for (int g = 0; g < 8; g++) {
        printf("%.2f, ", f1score[g]);
    }
    printf("\n");
}

Matrix createNewMatrix(unsigned int nrow, unsigned int ncol) {
    Matrix m;
    m.row = nrow;
    m.col = ncol;
    m.matrix = malloc(nrow * ncol * sizeof(float));
    setValueMatrix(&m, 0.0);
    return m;
}

Matrix openFile(char *filename) {
    printf("reading file %s ...\n", filename);
    FILE* stream = fopen(filename, "r");
    
    int row, col;
    countLines(filename, &row, &col);
    printf(">> read %d rows and %d cols\n", row, col);
    
    Matrix m = createNewMatrix(row, col);
    char line[100000];
    char *record;
    int i = 0;
    while (fgets(line, 100000, stream)) {
        int j = 0;
        record = strtok(line, ",");
        while (record != NULL) {
            float d;
            sscanf(record, "%f", &d);
            m.matrix[IX(i,j,col)] = d;
            record = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    printf("finished reading %s\n\n", filename);
    fclose(stream);
    return m;
}