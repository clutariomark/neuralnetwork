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
void outputScores(FILE *fout, int q, float precision[8], float recall[8], float f1score[8], double f1score_mean);
void printScores(int confusion[8][8], int tp[8], int fp[8], int fn[8], float precision[8], float recall[8], float f1score[8]);
Matrix createNewMatrix(unsigned int nrow, unsigned int ncol);
Matrix openFile(char *filename);

int main(int argc, char* argv[]) {    
    char *ptr;
    long run_ann = strtol(argv[1], &ptr, 10);
    long train = strtol(argv[2], &ptr, 10);
    long hidden1_nodes = strtol(argv[1], &ptr, 10);
    long hidden2_nodes = strtol(argv[2], &ptr, 10);
    long max_epoch = strtol(argv[3], &ptr, 10);
    float eta = strtof(argv[4], &ptr);
    float f1_threshold =  strtof(argv[5], &ptr);
    long modprint = strtol(argv[6], &ptr, 10);
    char *trainfilename = argv[7];
    char *validfilename = argv[8];
    char *testfilename = argv[9];

    clock_t start, diff, superstart, superend, trnstart, trnend, valstart, valend, tststart, tstend;
    superstart = clock();

    time_t current_time;
    char buffer[26];
    struct tm* tm_info;
    time(&current_time);
    tm_info = localtime(&current_time);

    strftime(buffer, 26, "%Y%m%d_%H%M%S", tm_info);

    char outfilename[200];
    sprintf(outfilename, "out_%s.txt", buffer);
    FILE *fstat = fopen(outfilename, "w");
    
    int run_ann = 1;
    if (run_ann == 1) {
        
        /* TRAINING PHASE */
        /* Read files */
        Matrix data_mat = openFile("train_set.csv");
        Matrix label_mat = openFile("train_label.csv");
        Matrix valid_mat = openFile("validation_set.csv");
        Matrix validlabel_mat = openFile("validation_label.csv");
        Matrix test_mat = openFile("test_set.csv");
        
        int N = data_mat.row;
        int F = data_mat.col;
        
        int input_nodes = F;
        int output_nodes = 8;
        
        int N_array[N];
        for (int i = 0; i < N; i++) {
            N_array[i] = i;
        }

        fprintf(fstat, "Network Architecture\n");
        fprintf(fstat, "2 Layers\n");
        fprintf(fstat, "input nodes: %d\n", input_nodes);
        fprintf(fstat, "output nodes: %d\n", output_nodes);
        fprintf(fstat, "hidden layer 1 nodes: %ld\n", hidden1_nodes);
        fprintf(fstat, "hidden layer 2 nodes: %ld\n", hidden2_nodes);
        fprintf(fstat, "learning rate: %f\n\n", eta);
        
        // Set weight and bias matrices
        Matrix weight_h1 = createNewMatrix(hidden1_nodes, input_nodes);
        Matrix weight_h1_temp = createNewMatrix(hidden1_nodes, input_nodes);
        Matrix weight_h2 = createNewMatrix(hidden2_nodes, hidden1_nodes);
        Matrix weight_h2_temp = createNewMatrix(hidden2_nodes, hidden1_nodes);
        Matrix weight_out = createNewMatrix(output_nodes, hidden2_nodes);
        Matrix weight_out_temp = createNewMatrix(output_nodes, hidden2_nodes);
        Matrix bias_h1 = createNewMatrix(hidden1_nodes, 1);
        Matrix bias_h2 = createNewMatrix(hidden2_nodes, 1);
        Matrix bias_out = createNewMatrix(output_nodes, 1);
        
        randomizeMatrix(&weight_h1);
        randomizeMatrix(&weight_h2);
        randomizeMatrix(&weight_out);
        randomizeMatrix(&bias_h1);
        randomizeMatrix(&bias_h2);
        randomizeMatrix(&bias_out);
        
        float num = (0.1 + 0.1);
        
        multiplyScalar(&weight_h1, num);
        addScalar(&weight_h1, -0.1);
        multiplyScalar(&weight_h2, num);
        addScalar(&weight_h2, -0.1);
        multiplyScalar(&weight_out, num);
        addScalar(&weight_out, -0.1);
        
        multiplyScalar(&bias_h1, num);
        addScalar(&bias_h1, -0.1);
        multiplyScalar(&bias_h2, num);
        addScalar(&bias_h2, -0.1);
        multiplyScalar(&bias_out, num);
        addScalar(&bias_out, -0.1);
        
        // Initialize Matrix holders
        Matrix x_in = createNewMatrix(F, 1);
        Matrix x_in_t = createNewMatrix(1, F);
        Matrix d_out = createNewMatrix(8, 1);
        Matrix v_h1 = createNewMatrix(weight_h1.row, x_in.col);
        Matrix y_h1 = createNewMatrix(v_h1.row, v_h1.col);
        Matrix y_h1_t = createNewMatrix(v_h1.col, v_h1.row);
        Matrix v_h2 = createNewMatrix(weight_h2.row, y_h1.col);
        Matrix y_h2 = createNewMatrix(v_h2.row, v_h2.col);
        Matrix y_h2_t = createNewMatrix(v_h2.col, v_h2.row);
        Matrix v_out = createNewMatrix(weight_out.row, y_h2.col);
        Matrix out = createNewMatrix(v_out.row, v_out.col);
        Matrix err = createNewMatrix(out.row, out.col);
        Matrix delta_out = createNewMatrix(err.row, err.col);
        Matrix delta_h2 = createNewMatrix(weight_out.col, delta_out.col);
        Matrix delta_h1 = createNewMatrix(weight_h2.col, delta_h2.col);
        
        int train = 1;
        if (train == 1) {
            printf("TRAINING PART\n");
            fprintf(fstat, "TRAINING PART\n");
            trnstart = clock();

            FILE *fout = fopen(trainfilename, "w");
            if (fout == NULL) {
                printf(">> Error opening file!\n");
                exit(1);
            }
            fprintf(fout, "epoch,p1,p2,p3,p4,p5,p6,p7,p8,r1,r2,r3,r4,r5,r6,r7,r8,f1,f2,f3,f4,f5,f6,f7,f8,f1score_mean\n");


            for (int q = 0; q < max_epoch; q++) {
                shuffleArray(N_array, N);
                float err_double = 0.0;
                float out_double = 0.0;
                int index;
                int unclassified = 0;
                int confusion[8][8] = {{0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0}};
                                       
                // Iterate through all the rows
                start = clock();
                for (int n = 0; n < N; n++) {
                    int nn = N_array[n];
                    /* read data */
                    setValueMatrix(&d_out, 0);
                    getSubMatrix(&x_in, data_mat, nn, nn+1, 0, F, 1);
                    index = label_mat.matrix[IX(nn, 0, label_mat.col)];
                    d_out.matrix[IX(index-1, 0, d_out.col)] = 1.00;
                    
                    /* forward pass */
                    // hidden layer 1
                    multiplyMatrix(&v_h1, weight_h1, x_in);
                    addSubtractMatrix(&v_h1, v_h1, bias_h1, 0);
                    computeY(&y_h1, v_h1);
                    
                    // hidden layer 2
                    multiplyMatrix(&v_h2, weight_h2, y_h1);
                    addSubtractMatrix(&v_h2, v_h2, bias_h2, 0);
                    computeY(&y_h2, v_h2);
                    
                    // output layer
                    multiplyMatrix(&v_out, weight_out, y_h2);
                    addSubtractMatrix(&v_out, v_out, bias_out, 0);
                    computeY(&out, v_out);
                    
                    int predicted_index = 10;
                    double outvalue = 0.0;
                    int actual_index = index - 1;
                    for (int m = 0; m < output_nodes; m++) {
                        double outlabel = out.matrix[IX(m,0,out.col)];
                        if (outlabel > outvalue) {
                            outvalue = outlabel;
                            predicted_index = m;
                        }
                    }
                    
                    if (predicted_index == 10) {
                        unclassified = unclassified + 1;
                    } else {
                        confusion[actual_index][predicted_index] = confusion[actual_index][predicted_index] + 1;
                    }
                    
                    /* error backpropagation */
                    // compute error
                    addSubtractMatrix(&err, d_out, out, 1);
                    err_double = err.matrix[IX(0,0,err.col)];
                    out_double = out.matrix[IX(0,0,out.col)];
                    
                    // compute gradient in output layer
                    computeDeltaOut(&delta_out, err, out);
                    
                    // compute gradient in hidden layer 2
                    computeDeltaHidden(&delta_h2, y_h2, weight_out, delta_out);
                    
                    // compute gradient in hidden layer 1
                    computeDeltaHidden(&delta_h1, y_h1, weight_h2, delta_h2);
                    
                    // update weights and biases in output layer
                    updateWeights(&weight_out, &weight_out_temp, delta_out, y_h2, &y_h2_t, eta);
                    updateBias(&bias_out, delta_out, eta);

                    // update weights and biases in hidden layer 2
                    updateWeights(&weight_h2, &weight_h2_temp, delta_h2, y_h1, &y_h1_t, eta);
                    updateBias(&bias_h2, delta_h2, eta);

                    // update weights and biases in hidden layer 1
                    updateWeights(&weight_h1, &weight_h1_temp, delta_h1, x_in, &x_in_t, eta);
                    updateBias(&bias_h1, delta_h1, eta);
                }

                /* calculate f-score from confusion matrix */
                int tp[8] = {0,0,0,0,0,0,0,0};
                int fp[8] = {0,0,0,0,0,0,0,0};
                int fn[8] = {0,0,0,0,0,0,0,0};
                float precision[8] = {0,0,0,0,0,0,0,0};
                float recall[8] = {0,0,0,0,0,0,0,0};
                float f1score[8] = {0,0,0,0,0,0,0,0};

                for(int d = 0; d < 8; d++) {
                    tp[d] = tp[d] + confusion[d][d];
                    for(int e = 0; e < 8; e++) {
                        if (d != e) {
                            fn[d] = fn[d] + confusion[d][e];
                            fp[e] = fp[e] + confusion[d][e];
                        }
                    }
                }

                for (int f = 0; f < 8; f++) {
                    precision[f] = (float)tp[f]/((float)tp[f] + (float)fp[f]);
                    recall[f] = (float)tp[f]/((float)tp[f] + (float)fn[f]);
                    f1score[f] = 2*precision[f]*recall[f]/(precision[f]+recall[f]);
                }

                double f1score_mean = 0;
                for (int g = 0; g < 8; g++) {
                    f1score_mean = f1score_mean + f1score[g];
                }
                f1score_mean = f1score_mean/8;

                diff = clock() - start;
                int msec = diff * 1000 / CLOCKS_PER_SEC;
                
                if (q % modprint == 0 || f1score_mean >= f1_threshold) {
                    printf("epoch no. %d\n", q);
                    printScores(confusion, tp, fp, fn, precision, recall, f1score);
                    printf("unclassified: %d\n", unclassified);
                    printf("f1 score mean: %.2f\n", f1score_mean);
                    printf("Time taken %d seconds %d milliseconds\n\n", msec/1000, msec%1000);
                }

                outputScores(fout, q, precision, recall, f1score, f1score_mean);

                if (f1score_mean >= f1_threshold) {
                    break;
                }
            }
            fclose(fout);
            trnend = clock() - trnstart;
            int msec = trnend * 1000 / CLOCKS_PER_SEC;
            fprintf(fstat, "Time taken %d seconds %d milliseconds\n\n", msec/1000, msec%1000);

        }
        
        int validationphase = 1;
        if (validationphase == 1) {
            /* VALIDATION PART */
            printf("VALIDATION PART\n");
            fprintf(fstat, "VALIDATION PART\n");
            valstart = clock();
            Matrix xv_in = createNewMatrix(F, 1);
            Matrix dv_out = createNewMatrix(8, 1);
            Matrix vv_h1_temp = createNewMatrix(weight_h1.row, xv_in.col);
            Matrix vv_h1 = createNewMatrix(weight_h1.row, xv_in.col);
            Matrix yv_h1 = createNewMatrix(vv_h1.row, vv_h1.col);
            Matrix vv_h2_temp = createNewMatrix(weight_h2.row, yv_h1.col);
            Matrix vv_h2 = createNewMatrix(weight_h2.row, yv_h1.col);
            Matrix yv_h2 = createNewMatrix(vv_h2.row, vv_h2.col);
            Matrix vv_out_temp = createNewMatrix(weight_out.row, yv_h2.col);
            Matrix vv_out = createNewMatrix(weight_out.row, yv_h2.col);
            Matrix out_v = createNewMatrix(vv_out.row, vv_out.col);
            
            int confusion[8][8] = {{0,0,0,0,0,0,0,0},
                                   {0,0,0,0,0,0,0,0},
                                   {0,0,0,0,0,0,0,0},
                                   {0,0,0,0,0,0,0,0},
                                   {0,0,0,0,0,0,0,0},
                                   {0,0,0,0,0,0,0,0},
                                   {0,0,0,0,0,0,0,0},
                                   {0,0,0,0,0,0,0,0}};

            int index_v;
            int unclassifiedv = 0;
            int N_train = valid_mat.row;
            FILE *fval = fopen(validfilename, "w");
            for (int p = 0; p < N_train; p++) {
                
                /* read matrix */
                getSubMatrix(&xv_in, valid_mat, p, p+1, 0, F, 0);
                index_v = validlabel_mat.matrix[IX(p, 0, validlabel_mat.col)];
                dv_out.matrix[IX(index_v-1, 0, dv_out.col)] = 1.00;
                
                /* forward pass */
                // hidden layer 1
                multiplyMatrix(&vv_h1_temp, weight_h1, xv_in);
                addSubtractMatrix(&vv_h1, vv_h1_temp, bias_h1, 0);
                computeY(&yv_h1, vv_h1);
                
                // hidden layer 2
                multiplyMatrix(&vv_h2_temp, weight_h2, yv_h1);
                addSubtractMatrix(&vv_h2, vv_h2_temp, bias_h2, 0);
                computeY(&yv_h2, vv_h2);
                
                // output layer
                multiplyMatrix(&vv_out_temp, weight_out, yv_h2);
                addSubtractMatrix(&vv_out, vv_out_temp, bias_out, 0);
                computeY(&out_v, vv_out);
                
                int valid_out = 10;
                double outvalue = 0.0;
                for (int m = 0; m < output_nodes; m++) {
                    double outlabel = out_v.matrix[IX(m,0,out_v.col)];
                    if (outlabel > outvalue) {
                        outvalue = outlabel;
                        valid_out = m + 1;
                    }
                }

                for (int m = 0; m < output_nodes; m++) {
                    fprintf(fval, "%.2f ", out_v.matrix[IX(m,0,out_v.col)]);
                }
                fprintf(fval, "-- %d|%d\n", valid_out, index_v);

                if (valid_out == 10) {
                    unclassifiedv = unclassifiedv + 1;
                } else {
                    confusion[index_v-1][valid_out-1] = confusion[index_v-1][valid_out-1] + 1;
                }
                
            }

            /* calculate f-score from confusion matrix */
            int tp[8] = {0,0,0,0,0,0,0,0};
            int fp[8] = {0,0,0,0,0,0,0,0};
            int fn[8] = {0,0,0,0,0,0,0,0};
            float precision[8] = {0,0,0,0,0,0,0,0};
            float recall[8] = {0,0,0,0,0,0,0,0};
            float f1score[8] = {0,0,0,0,0,0,0,0};

            for(int d = 0; d < 8; d++) {
                tp[d] = tp[d] + confusion[d][d];
                for(int e = 0; e < 8; e++) {
                    if (d != e) {
                        fn[d] = fn[d] + confusion[d][e];
                        fp[e] = fp[e] + confusion[d][e];
                    }
                }
            }

            for (int f = 0; f < 8; f++) {
                precision[f] = (float)tp[f]/((float)tp[f] + (float)fp[f]);
                recall[f] = (float)tp[f]/((float)tp[f] + (float)fn[f]);
                f1score[f] = 2*precision[f]*recall[f]/(precision[f]+recall[f]);
            }

            double f1score_mean = 0;
            for (int g = 0; g < 8; g++) {
                f1score_mean = f1score_mean + f1score[g];
            }
            f1score_mean = f1score_mean/8;

            printScores(confusion, tp, fp, fn, precision, recall, f1score);
            printf("unclassified: %d\n", unclassifiedv);
            printf("f1 score mean: %.2f\n", f1score_mean);
            fprintf(fval, "\n");
            fprintf(fval, "epoch,p1,p2,p3,p4,p5,p6,p7,p8,r1,r2,r3,r4,r5,r6,r7,r8,f1,f2,f3,f4,f5,f6,f7,f8,f1score_mean\n");
            outputScores(fval, 0, precision, recall, f1score, f1score_mean);

            fclose(fval);
            valend = clock() - valstart;
            int msec = valend * 1000 / CLOCKS_PER_SEC;
            fprintf(fstat, "Time taken %d seconds %d milliseconds\n\n", msec/1000, msec%1000);
        }

        int testphase = 1;
        if (testphase == 1) {
            /* TEST PHASE */
            printf("TEST PART\n");
            fprintf(fstat, "TEST PART\n");
            tststart = clock();

            FILE *ftest = fopen(testfilename, "w");
            
            Matrix xt_in = createNewMatrix(F, 1);
            Matrix vt_h1_temp = createNewMatrix(weight_h1.row, xt_in.col);
            Matrix vt_h1 = createNewMatrix(weight_h1.row, xt_in.col);
            Matrix yt_h1 = createNewMatrix(vt_h1.row, vt_h1.col);
            Matrix vt_h2_temp = createNewMatrix(weight_h2.row, yt_h1.col);
            Matrix vt_h2 = createNewMatrix(weight_h2.row, yt_h1.col);
            Matrix yt_h2 = createNewMatrix(vt_h2.row, vt_h2.col);
            Matrix vt_out_temp = createNewMatrix(weight_out.row, yt_h2.col);
            Matrix vt_out = createNewMatrix(weight_out.row, yt_h2.col);
            Matrix out_t = createNewMatrix(vt_out.row, vt_out.col);
            
            int N_train = test_mat.row;
            for (int p = 0; p < N_train; p++) {
                /* read matrix */
                
                getSubMatrix(&xt_in, test_mat, p, p+1, 0, F, 0);
                
                /* forward pass */
                // hidden layer 1
                multiplyMatrix(&vt_h1_temp, weight_h1, xt_in);
                addSubtractMatrix(&vt_h1, vt_h1_temp, bias_h1, 0);
                computeY(&yt_h1, vt_h1);
                
                // hidden layer 2
                multiplyMatrix(&vt_h2_temp, weight_h2, yt_h1);
                addSubtractMatrix(&vt_h2, vt_h2_temp, bias_h2, 0);
                computeY(&yt_h2, vt_h2);
                
                // output layer
                multiplyMatrix(&vt_out_temp, weight_out, yt_h2);
                addSubtractMatrix(&vt_out, vt_out_temp, bias_out, 0);
                computeY(&out_t, vt_out);
                
                int test_out = 0;
                double outvalue = 0.0;
                for (int m = 0; m < output_nodes; m++) {
                    double outlabel = out_t.matrix[IX(m,0,out_t.col)];
                    if (outlabel > outvalue) {
                        outvalue = outlabel;
                        test_out = m + 1;
                    }
                }

                for (int m = 0; m < output_nodes; m++) {
                    fprintf(ftest, "%.2f ", out_t.matrix[IX(m,0,out_t.col)]);
                }
                fprintf(ftest, "-- %d\n", test_out);
                // printf("%3d) test_out: %d\n", p, test_out);
                
            }

            fclose(ftest);
            tstend = clock() - tststart;
            int msec = tstend * 1000 / CLOCKS_PER_SEC;
            fprintf(fstat, "Time taken %d seconds %d milliseconds\n\n", msec/1000, msec%1000);
        }
    }
    
    superend = clock() - superstart;
    int msec = superend * 1000/ CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

    fclose(fstat);
    
    printf("Press Enter to continue...");
    getchar();
    
    return 0;
    
}

void setValueMatrix(Matrix *m, float value) {
    
    //! Set all elements of matrix to be equal a certain value
    /*! 
        \param m the input matrix
        \param value the value of the matrix to be set to    
    */

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

void outputScores(FILE *fout, int q, float precision[8], float recall[8], float f1score[8], double f1score_mean) {
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
    fprintf(fout, "%f\n", f1score_mean);
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