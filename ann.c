#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "functions.h"


int main(int argc, char* argv[]) {    
    char *ptr;
    // long run_ann = strtol(argv[1], &ptr, 10);
    // long train = strtol(argv[2], &ptr, 10);
    long hidden1_nodes = strtol(argv[1], &ptr, 10);
    long hidden2_nodes = strtol(argv[2], &ptr, 10);
    long max_epoch = strtol(argv[3], &ptr, 10);
    float eta = strtof(argv[4], &ptr);
    float f1_threshold =  strtof(argv[5], &ptr);
    long modprint = strtol(argv[6], &ptr, 10);
    // char *trainfilename = argv[7];
    // char *validfilename = argv[8];
    // char *testfilename = argv[9];

    clock_t start, diff, superstart, superend, trnstart, trnend, tststart, tstend;
    superstart = clock();

    time_t current_time;
    char buffer[26];
    struct tm* tm_info;
    time(&current_time);
    tm_info = localtime(&current_time);

    strftime(buffer, 26, "%Y%m%d_%H%M%S", tm_info);

    char outfilename[200];
    char trainfilename[200];
    char validfilename[200];
    char testfilename[200];
    sprintf(outfilename, "runstatus_%s_H%ldH%ldLR%.f.txt", buffer, hidden1_nodes, hidden2_nodes, eta*100);
    sprintf(trainfilename, "train_H%ldH%ldLR%.f.csv", hidden1_nodes, hidden2_nodes, eta*100);
    sprintf(validfilename, "valid_H%ldH%ldLR%.f.csv", hidden1_nodes, hidden2_nodes, eta*100);
    sprintf(testfilename, "test_H%ldH%ldLR0%.f.txt", hidden1_nodes, hidden2_nodes, eta*100);
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

        int Nv = valid_mat.row;
        
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
        // Training Part
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

        // Validation Part
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
        
        int trainphase = 1;
        if (trainphase == 1) {
            printf("TRAINING PART\n");
            fprintf(fstat, "TRAINING PART\n");
            trnstart = clock();

            FILE *fout = fopen(trainfilename, "w");
            if (fout == NULL) {
                printf(">> Error opening training output file!\n");
                exit(1);
            }
            fprintf(fout, "epoch,p1,p2,p3,p4,p5,p6,p7,p8,r1,r2,r3,r4,r5,r6,r7,r8,f1,f2,f3,f4,f5,f6,f7,f8,f1score_mean,error_rate\n");

            FILE *fval = fopen(validfilename, "w");
            if (fval == NULL) {
                printf(">> Error opening validation output file!\n");
                exit(1);
            }
            fprintf(fval, "epoch,p1,p2,p3,p4,p5,p6,p7,p8,r1,r2,r3,r4,r5,r6,r7,r8,f1,f2,f3,f4,f5,f6,f7,f8,f1score_mean,error_rate\n");

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

                int confusion_v[8][8] = {{0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0},
                                       {0,0,0,0,0,0,0,0}};

                double error = 0;
                                       
                // Training: Updating weights and biases
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

                    if (actual_index != predicted_index) error = error + 1;
                    
                    /* error backpropagation */
                    // compute error
                    addSubtractMatrix(&err, d_out, out, 1);
                    
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

                // Test trained network in Validation Samples
                int index_v;
                int unclassifiedv = 0;
                int N_train = valid_mat.row;
                double errorv = 0.0;
                
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

                    if (valid_out == 10) {
                        unclassifiedv = unclassifiedv + 1;
                    } else {
                        confusion_v[index_v-1][valid_out-1] = confusion_v[index_v-1][valid_out-1] + 1;
                    }

                    if (index_v != valid_out) errorv = errorv + 1;
                    
                }

                /* calculate f-score from confusion matrix */
                int tp[8] = {0,0,0,0,0,0,0,0};
                int fp[8] = {0,0,0,0,0,0,0,0};
                int fn[8] = {0,0,0,0,0,0,0,0};
                float precision[8] = {0,0,0,0,0,0,0,0};
                float recall[8] = {0,0,0,0,0,0,0,0};
                float f1score[8] = {0,0,0,0,0,0,0,0};

                int tpv[8] = {0,0,0,0,0,0,0,0};
                int fpv[8] = {0,0,0,0,0,0,0,0};
                int fnv[8] = {0,0,0,0,0,0,0,0};
                float precisionv[8] = {0,0,0,0,0,0,0,0};
                float recallv[8] = {0,0,0,0,0,0,0,0};
                float f1scorev[8] = {0,0,0,0,0,0,0,0};

                for(int d = 0; d < 8; d++) {
                    tp[d] = tp[d] + confusion[d][d];
                    tpv[d] = tpv[d] + confusion_v[d][d];
                    for(int e = 0; e < 8; e++) {
                        if (d != e) {
                            fn[d] = fn[d] + confusion[d][e];
                            fp[e] = fp[e] + confusion[d][e];
                            fnv[d] = fnv[d] + confusion_v[d][e];
                            fpv[e] = fpv[e] + confusion_v[d][e];
                        }
                    }
                }

                for (int f = 0; f < 8; f++) {
                    precision[f] = (float)tp[f]/((float)tp[f] + (float)fp[f]);
                    recall[f] = (float)tp[f]/((float)tp[f] + (float)fn[f]);
                    f1score[f] = 2*precision[f]*recall[f]/(precision[f]+recall[f]);

                    precisionv[f] = (float)tpv[f]/((float)tpv[f] + (float)fpv[f]);
                    recallv[f] = (float)tpv[f]/((float)tpv[f] + (float)fnv[f]);
                    f1scorev[f] = 2*precisionv[f]*recallv[f]/(precisionv[f]+recallv[f]);
                }

                double f1score_mean = 0;
                double f1score_meanv = 0;
                for (int g = 0; g < 8; g++) {
                    f1score_mean = f1score_mean + f1score[g];
                    f1score_meanv = f1score_meanv + f1scorev[g];
                }

                f1score_mean = f1score_mean/8;
                f1score_meanv = f1score_meanv/8;

                double mse = error/N;
                double mse_v = errorv/Nv;

                diff = clock() - start;
                int msec = diff * 1000 / CLOCKS_PER_SEC;
                
                if (q % modprint == 0 || f1score_mean >= f1_threshold) {
                    printf("epoch no. %d\n", q);

                    printf("TRAINING RESULTS:\n");
                    printScores(confusion, tp, fp, fn, precision, recall, f1score);
                    printf("f1 score mean: %.2f\n", f1score_mean);
                    printf("error rate: %.7f\n", mse);
                    printf("********************************\n");
                    printf("VALIDATION RESULTS:\n");
                    printScores(confusion_v, tpv, fpv, fnv, precisionv, recallv, f1scorev);
                    printf("f1 score mean: %.2f\n", f1score_meanv);
                    printf("error rate: %.7f\n", mse_v);
                    printf("Time taken %d seconds %d milliseconds\n\n", msec/1000, msec%1000);
                }

                outputScores(fout, q, precision, recall, f1score, f1score_mean, mse);
                outputScores(fval, q, precisionv, recallv, f1scorev, f1score_meanv, mse_v);

                if (f1score_mean >= f1_threshold) {
                    break;
                }
            }

            fclose(fout);
            fclose(fval);
            trnend = clock() - trnstart;
            int msec = trnend * 1000 / CLOCKS_PER_SEC;
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
    
    // printf("Press Enter to continue...");
    // getchar();
    
    return 0;
    
}