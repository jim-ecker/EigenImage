package edu.gatech.jecker;

/**
 * Created by jecker on 6/10/16.
 */
public class PCA {

    // The incoming matrix
    private final Matrix m;
    // the principal components
    private final Matrix pc;
    // facpr
    private final Matrix facpr;
    // lambda
    private final Matrix lambda;

    public PCA(Matrix x) {

        // Weight and center the matrix
        this.m = x.weightedCenter();
        // compute the eigenvectors of y'*y using svd
        SingleValueDecomposition svd = new SingleValueDecomposition(this.m);

        // calculate the lambda
        this.lambda = calculateLambda(svd.getS());
        // get the principle factors
        this.facpr = svd.getV();

        // calculate the principle components
        this.pc = this.m.times(svd.getV());
    }

    private Matrix calculateLambda(Matrix s) {

        Matrix d = s.getDiagonal();
        double[][] D = d.getArray();

        int size = d.getNumRows();
        for (int i = 0; i < size; i++) {
            D[i][0] = (D[i][0] * D[i][0]) / (size - 1);
        }

        return d;
    }

    public Matrix getPrincipalComponents() {
        return pc;
    }

    public Matrix getLambda() {
        return lambda;
    }

    public Matrix getPrinicipalFactors() {
        return facpr;
    }

}