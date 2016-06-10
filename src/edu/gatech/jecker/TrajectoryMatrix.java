package edu.gatech.jecker;

/**
 * Created by jecker on 6/10/16.
 */
public class TrajectoryMatrix extends Matrix {

    public TrajectoryMatrix(double[] v, int ncols) {
        super(v.length - ncols + 1, ncols);
        double[][] arr = getArray();
        int nrows = getNumRows();
        int pos = 0; // position in vector

        for (int i = 0; i < nrows; i++) {
            double value = v[pos++];
            int availCols = i < ncols ? i + 1 : ncols;
            for (int j = 0, m = i; j < availCols && m >= 0; j++, m--) {
                arr[m][j] = value;
            }
        }
        for (int i = 1; i < ncols; i++) {
            double value = v[pos++];
            for (int j = i, m = nrows - 1; j < ncols && m > 0; j++, m--) {
                arr[m][j] = value;
            }
        }
    }
}