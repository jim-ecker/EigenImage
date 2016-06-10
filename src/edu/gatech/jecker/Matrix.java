package edu.gatech.jecker;

import java.util.ArrayList;

/**
 * The Matrix class provides an API for representing
 * a matrix (numerical data organized in two dimensions)
 * and performing Linear Algebraic operations on it.
 *
 * Created by Jim Ecker on 6/10/16.
 */
public class Matrix {

    // Array for internal storage of elements.
    private final double[][] matrix;
    // Row and column dimensions.
    private final int rows, cols;

    /**
     * Construct an m-by-n matrix of zeros.
     *
     * @param rows Number of rows.
     * @param cols Number of colums.
     */
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.matrix = new double[this.rows][this.cols];
    }

    /**
     * Construct an m-by-n constant matrix.
     *
     * @param rows  Number of rows.
     * @param cols  Number of colums.
     * @param fill  Fill the matrix with this scalar value.
     */
    public Matrix(int rows, int cols, double fill) {
        this.rows = rows;
        this.cols = cols;
        this.matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.matrix[i][j] = fill;
            }
        }
    }

    /**
     * Construct a matrix from a 2-D array.
     *
     * @param matrix    Two-dimensional array of doubles.
     */
    public Matrix(double[][] matrix) {
        rows = matrix.length;
        cols = matrix[0].length;
        for (int i = 0; i < rows; i++) {
            if (matrix[i].length != cols) {
                throw new MatrixException("All rows must have the same length.");
            }
        }
        this.matrix = matrix;
    }

    /**
     * Create a matrix hold-my-beer style (don't validate)
     *
     * @param matrix Two-dimensional array of doubles.
     * @param rows Number of rows.
     * @param cols Number of colums.
     */
    public Matrix(double[][] matrix, int rows, int cols) {
        this.matrix = matrix;
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Create a matrix from a one-dimensional packed array
     *
     * @param vals One-dimensional array of doubles, column-wise packed
     * @param rows Number of rows.
     */
    public Matrix(double vals[], int rows) {
        this.rows = rows;
        cols = (rows != 0 ? vals.length / rows : 0);
        if(rows * cols != vals.length)
            throw new MatrixException("Packed array must be square. Array length not a multiple of rows");
        this.matrix = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                this.matrix[i][j] = vals[i + j * rows];
            }
        }
    }

    /**
     * Center the matrix values
     *
     * @return this.matrix, centered
     */
    public Matrix center() {

        Matrix centered = new Matrix(this.rows, this.cols);
        double[][] vals = centered.getArray();
        for(int i = 0; i < rows; i++) {
            double sum = 0;
            for(int j = 0; j < cols; j++) {
                sum += this.matrix[i][j];
            }
            double mean = sum / cols;
            for(int j = 0; j < cols; j++) {
                vals[i][j] = this.matrix[i][j] - mean;
            }
        }
        return centered;
    }

    /**
     * Center and weight Sample variance for grouped data
     *
     * s^2 = SUM(Mi - xbar)^2 / (n-1)
     *
     * @return this.matrix weighted and centered
     */
    public Matrix weightedCenter() {
        // center the data
        Matrix weighted = center().transpose();
        double[][] vals = weighted.getArray();

        // calculate the sample variance
        double[] sigma = new double[this.cols];
        for (int j = 0; j < cols; j++) {
            double sigmaSum = 0;
            for (int i = 0; i < rows; i++) {
                sigmaSum += (vals[i][j] * vals[i][j]);
            }
            sigma[j] = Math.sqrt(sigmaSum / (this.rows- 1));
        }

        // weigh the data
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                vals[i][j] = vals[i][j] / sigma[j];
            }
        }

        return weighted;
    }

    /**
     * Get the diagonal as a 1-col matrix
     *
     * @return diagonal
     */
    public Matrix getDiagonal() {

        int numRows = Math.min(this.rows, this.cols);
        Matrix diagonal = new Matrix(numRows, 1);
        double[][] vals = diagonal.getArray();

        for (int i = 0; i < numRows; i++) {
            vals[i][0] = this.matrix[i][i];
        }

        return diagonal;
    }

    /**
     * Access the internal two-dimensional array.
     *
     * @return Pointer to the two-dimensional array of matrix elements.
     */
    public double[][] getArray() {
        return this.matrix;
    }

    /**
     * Copy the internal two-dimensional array as value.
     *
     * @return Two-dimensional array copy of matrix elements.
     */
    public double[][] getArrayCopy() {
        double[][] copy = new double[this.rows][this.cols];
        for (int i = 0; i < this.rows; i++) {
            System.arraycopy(this.matrix[i], 0, copy[i], 0, this.cols);
        }
        return copy;
    }

    /**
     * Get row dimension.
     *
     * @return this.rows, the number of rows.
     */
    public int getNumRows() {
        return this.rows;
    }

    /**
     * Get column dimension.
     *
     * @return this.cols, the number of columns.
     */
    public int getNumCols() {
        return this.cols;
    }

    /**
     * Get a single element.
     *
     * @param i Row index.
     * @param j Column index.
     * @return this.matrix(i,j)
     */
    public double get(int i, int j) {
        return this.matrix[i][j];
    }

    /**
     * Get a submatrix.
     *
     * @param fromRow Initial row index
     * @param toRow Final row index
     * @param fromCol Initial column index
     * @param toCol Final column index
     * @return this.matrix(fromRow:toRow,fromCol:toCol)
     */
    public Matrix getSubMatrix(int fromRow, int toRow, int fromCol, int toCol) {
        Matrix sub = new Matrix(toRow - fromRow + 1, toCol - fromCol + 1);
        double[][] vals = sub.getArray();
        for(int i = fromRow; i <= toRow; i++) {
            for(int j = fromCol; j <= toCol; j++) {
                vals[i - fromRow][j - fromCol] = this.matrix[i][j];
            }
        }
        return sub;
    }

    /**
     * Get a submatrix.
     *
     * @param rows Array of row indices.
     * @param cols Array of column indices.
     * @return this.matrix(rows(:), cols(:))
     */
    public Matrix getSubMatrix(int[] rows, int[] cols) {
        Matrix sub = new Matrix(rows.length, cols.length);
        double[][] vals = sub.getArray();
        for(int i = 0; i < rows.length; i++) {
            for(int j = 0; j < cols.length; j++) {
                vals[i][j] = this.matrix[rows[i]][cols[j]];
            }
        }
        return sub;
    }

    /**
     * Get a submatrix.
     *
     * @param fromRow Initial row index
     * @param toRow Final row index
     * @param cols Array of column indices.
     * @return this.matrix(fromRow:toRow,cols(:))
     */
    public Matrix getSubMatrix(int fromRow, int toRow, int[] cols) {
        Matrix sub = new Matrix(toRow - fromRow + 1, cols.length);
        double[][] vals = sub.getArray();
        for (int i = fromRow; i <= toRow; i++) {
            for (int j = 0; j < cols.length; j++) {
                vals[i - toRow][j] = this.matrix[i][cols[j]];
            }
        }
        return sub;
    }

    /**
     * Get a submatrix.
     *
     * @param rows Array of row indices.
     * @param fromCol Initial column index
     * @param toCol Final column index
     * @return this.matrix(rows(:),fromCol:toCol)
     */
    public Matrix getSubMatrix(int[] rows, int fromCol, int toCol) {
        Matrix sub = new Matrix(rows.length, toCol - fromCol + 1);
        double[][] vals = sub.getArray();
        for (int i = 0; i < rows.length; i++) {
            for (int j = fromCol; j <= toCol; j++) {
                vals[i][j - fromCol] = this.matrix[rows[i]][j];
            }
        }
        return sub;
    }

    /**
     * Set a single element in this Matrix.
     *
     * @param i Row index.
     * @param j Column index.
     * @param val this.matrix(i,j).
     */
    public void set(int i, int j, double val) {
        this.matrix[i][j] = val;
    }

    /**
     * Set a submatrix.
     *
     * @param fromRow Initial row index
     * @param toRow Final row index
     * @param fromCol Initial column index
     * @param toCol Final column index
     * @param insert this.matrix(fromRow:toRow,fromCol:toCol)
     */
    public void setSubMatrix(int fromRow, int toRow, int fromCol, int toCol, Matrix insert) {
        for(int i = fromRow; i <= toRow; i++) {
            for(int j = fromCol; j <= toCol; j++) {
                this.matrix[i][j] = insert.get(i - fromRow, j - fromCol);
            }
        }
    }

    /**
     * Set a submatrix.
     *
     * @param rows Array of row indices.
     * @param cols Array of column indices.
     * @param insert this.matrix(rows(:),cols(:))
     */
    public void setSubMatrix(int[] rows, int[] cols, Matrix insert) {
        for(int i = 0; i <= rows.length; i++) {
            for(int j = 0; j <= cols.length; j++) {
                this.matrix[rows[i]][cols[j]] = insert.get(i, j);
            }
        }
    }

    /**
     * Set a submatrix.
     *
     * @param rows Array of row indices.
     * @param fromCol Initial column index
     * @param toCol Final column index
     * @param insert this.matrix(rows(:),fromCol:toCol)
     */
    public void setMatrix(int[] rows, int fromCol, int toCol, Matrix insert) {
        for (int i = 0; i < rows.length; i++) {
            for (int j = fromCol; j <= toCol; j++) {
                this.matrix[rows[i]][j] = insert.get(i, j - fromCol);
            }
        }
    }

    /**
     * Set a submatrix.
     *
     * @param fromRow Initial row index
     * @param toRow Final row index
     * @param cols Array of column indices.
     * @param insert this.matrix(fromRow:toRow,cols(:))
     */
    public void setMatrix(int fromRow, int toRow, int[] cols, Matrix insert) {
        for (int i = fromRow; i <= toRow; i++) {
            for (int j = 0; j < cols.length; j++) {
                this.matrix[i][cols[j]] = insert.get(i - fromRow, j);
            }
        }
    }

    /**
     * Matrix transpose.
     *
     * @return this.matrix'
     */
    public Matrix transpose() {
        Matrix transposed = new Matrix(this.cols, this.rows);
        double[][] vals = transposed.getArray();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                vals[j][i] = this.matrix[i][j];
            }
        }
        return transposed;
    }

    /**
     * One norm
     *
     * @return maximum column sum.
     */
    public double oneNorm() {
        double maxOneNorm = 0;
        for (int j = 0; j < this.cols; j++) {
            double sum = 0;
            for (int i = 0; i < rows; i++) {
                sum += Math.abs(this.matrix[i][j]);
            }
            maxOneNorm = Math.max(maxOneNorm, sum);
        }
        return maxOneNorm;
    }

    /**
     * Two norm
     *
     * @return maximum singular value.
     */
    public double twoNorm() {
        return (new SingleValueDecomposition(this).twoNorm());
    }

    /**
     * Infinity norm
     *
     * @return maximum row sum.
     */
    public double infinityNorm() {
        double maxRowSum = 0;
        for (int i = 0; i < this.rows; i++) {
            double sum = 0;
            for (int j = 0; j < this.cols; j++) {
                sum += Math.abs(this.matrix[i][j]);
            }
            maxRowSum = Math.max(maxRowSum, sum);
        }
        return maxRowSum;
    }

    /**
     * Frobenius norm
     *
     * @return sqrt of sum of squares of all elements.
     */
    public double frobeniusNorm() {
        double frob = 0;
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                frob = Math.hypot(frob, this.matrix[i][j]);
            }
        }
        return frob;
    }

    /**
     * Unary minus
     *
     * @return -this.matrix
     */
    public Matrix unaryMinus() {
        Matrix subtracted = new Matrix(this.rows, this.cols);
        double[][] vals = subtracted.getArray();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                vals[i][j] = -this.matrix[i][j];
            }
        }
        return subtracted;
    }

    /**
     * C = this.matrix + B
     *
     * @param B another matrix
     * @return this.matrix + B
     */
    public Matrix plus(Matrix B) {
        checkMatrixDimensions(B);
        Matrix added = new Matrix(this.rows, this.cols);
        double[][] vals = added.getArray();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                vals[i][j] = this.matrix[i][j] + B.matrix[i][j];
            }
        }
        return added    ;
    }

    /**
     * A = this.matrix + B
     *
     * @param B another matrix
     * @return this.matrix + B
     */
    public Matrix plusEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.matrix[i][j] = this.matrix[i][j] + B.matrix[i][j];
            }
        }
        return this;
    }

    /**
     * C = this.matrix - B
     *
     * @param B another matrix
     * @return this.matrix - B
     */
    public Matrix minus(Matrix B) {
        checkMatrixDimensions(B);
        Matrix subtracted = new Matrix(this.rows, this.cols);
        double[][] vals = subtracted.getArray();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                vals[i][j] = this.matrix[i][j] - B.matrix[i][j];
            }
        }
        return subtracted;
    }

    /**
     * this.matrix = this.matrix - B
     *
     * @param B another matrix
     * @return this.matrix - B
     */
    public Matrix minusEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.matrix[i][j] = this.matrix[i][j] - B.matrix[i][j];
            }
        }
        return this;
    }

    /**
     * Element-wise multiplication, C = this.matrix.*B
     *
     * @param B another matrix
     * @return this.matrix.*B
     */
    public Matrix arrayTimes(Matrix B) {
        checkMatrixDimensions(B);
        Matrix multiplied = new Matrix(this.rows, this.cols);
        double[][] vals = multiplied.getArray();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                vals[i][j] = this.matrix[i][j] * B.matrix[i][j];
            }
        }
        return multiplied;
    }

    /**
     * Element-wise multiplication in place, this.matrix = this.matrix.*B
     *
     * @param B another matrix
     * @return this.matrix.*B
     */
    public Matrix arrayTimesEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.matrix[i][j] = this.matrix[i][j] * B.matrix[i][j];
            }
        }
        return this;
    }

    /**
     * Element-wise right division, C = this.matrix./B
     *
     * @param B another matrix
     * @return this.matrix./B
     */
    public Matrix arrayRightDivide(Matrix B) {
        checkMatrixDimensions(B);
        Matrix divided = new Matrix(this.rows, this.cols);
        double[][] vals = divided.getArray();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                vals[i][j] = this.matrix[i][j] / B.matrix[i][j];
            }
        }
        return divided;
    }

    /**
     * Element-wise right division in place, this.matrix = this.matrix./B
     *
     * @param B another matrix
     * @return this.matrix./B
     */
    public Matrix arrayRightDivideEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.matrix[i][j] = this.matrix[i][j] / B.matrix[i][j];
            }
        }
        return this;
    }

    /**
     * Element-wise left division, C = this.matrix.\B
     *
     * @param B another matrix
     * @return this.matrix.\B
     */
    public Matrix arrayLeftDivide(Matrix B) {
        checkMatrixDimensions(B);
        Matrix divided = new Matrix(this.rows, this.cols);
        double[][] vals = divided.getArray();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                vals[i][j] = B.matrix[i][j] / this.matrix[i][j];
            }
        }
        return divided;
    }

    /**
     * Element-wise left division in place, this.matrix = this.matrix.\B
     *
     * @param B another matrix
     * @return this.matrix.\B
     */
    public Matrix arrayLeftDivideEquals(Matrix B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.matrix[i][j] = B.matrix[i][j] / this.matrix[i][j];
            }
        }
        return this;
    }

    /**
     * Multiply a matrix by a scalar, C = scalar * this.matrix
     *
     * @param scalar scalar
     * @return scalar * this.matrix
     */
    public Matrix times(double scalar) {
        Matrix multiplied = new Matrix(this.rows, this.cols);
        double[][] vals = multiplied.getArray();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                vals[i][j] = scalar * this.matrix[i][j];
            }
        }
        return multiplied;
    }

    /**
     * Multiply a matrix by a scalar in place, this.matrix = scalar * this.matrix
     *
     * @param scalar scalar
     * @return replace this.matrix by scalar * this.matrix
     */
    public Matrix timesEquals(double scalar) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.matrix[i][j] = scalar * this.matrix[i][j];
            }
        }
        return this;
    }

    /**
     * Linear algebraic matrix multiplication, this.matrix * B
     *
     * @param B another matrix
     * @return Matrix product, this.matrix * B
     */
    public Matrix times(Matrix B) {
        if (B.getNumRows() != this.rows) {
            throw new MatrixException("Matrix inner dimensions must agree.");
        }
        Matrix multiplied = new Matrix(this.rows, B.getNumCols());
        double[][] vals = multiplied.getArray();
        double[] Bcolj = new double[this.cols];
        for (int j = 0; j < B.getNumCols(); j++) {
            for (int k = 0; k < cols; k++) {
                Bcolj[k] = B.matrix[k][j];
            }
            for (int i = 0; i < this.rows; i++) {
                double[] Arowi = this.matrix[i];
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += Arowi[k] * Bcolj[k];
                }
                vals[i][j] = sum;
            }
        }
        return multiplied;
    }

    /**
     * Matrix rank
     *
     * @return effective numerical rank, obtained from SingleValueDecomposition.
     */
    public int rank() {
        return new SingleValueDecomposition(this).rank();
    }

    /**
     * Matrix condition (2 norm)
     *
     * @return ratio of largest to smallest singular value.
     */
    public double cond() {
        return new SingleValueDecomposition(this).cond();
    }

    /**
     * Matrix trace.
     *
     * @return sum of the diagonal elements.
     */
    public double trace() {
        double trace = 0;
        for (int i = 0; i < Math.min(this.rows, this.cols); i++) {
            trace += this.matrix[i][i];
        }
        return trace;
    }

    /**
     * Generate matrix with random elements
     *
     * @param rows Number of rows.
     * @param cols Number of colums.
     * @return An m(rows)-by-n(cols) matrix with uniformly distributed random elements.
     */
    public static Matrix random(int rows, int cols) {
        Matrix random = new Matrix(rows, cols);
        double[][] X = random.getArray();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                X[i][j] = Math.random();
            }
        }
        return random;
    }

    /**
     * Generate identity matrix
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return An m(rows)-by-n(cols) matrix with ones on the diagonal and zeros elsewhere.
     */
    public static Matrix identity(int rows, int cols) {
        Matrix identity = new Matrix(rows, cols);
        double[][] X = identity.getArray();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                X[i][j] = (i == j ? 1.0 : 0.0);
            }
        }
        return identity;
    }

    /**
     * Check if size(this.matrix) == size(B)
     */
    private void checkMatrixDimensions(Matrix B) {
        if (B.getNumRows() != this.rows || B.getNumCols() != this.cols) {
            throw new IllegalArgumentException("Matrix dimensions must agree.");
        }
    }

    @Override
    public String toString() {
        return "Matrix(" + " Shape: " + this.rows + ", " + this.cols + ')';
    }

    private class Row {

        ArrayList<Double> vals;
        //double [] vals;

        Row() {
            this.vals = new ArrayList<>();
        }

        Row(double[] vals) {
            for(double val : vals)
                this.vals.add(val);
        }

        void append(double val) {
            this.vals.add(val);
        }

        void fill(double fill) {
            for(int i = 0; i < size(); i++) {
                this.vals.set(i, fill);
            }
        }

        int size() {
            return this.vals.size();
        }

        double sum() {
            double sum = 0;
            for(Double val : this.vals) {
                sum += val;
            }
            return sum;
        }
    }

    private class Column {

        ArrayList<Double> vals;
        //double [] vals;

        Column() {
            this.vals = new ArrayList<>();
        }

        Column(double[] vals) {
            for(double val : vals)
                this.vals.add(val);
        }

        protected void append(double val) {
            this.vals.add(val);
        }

        void fill(double fill) {
            for(int i = 0; i < size(); i++) {
                this.vals.set(i, fill);
            }
        }

        int size() {
            return this.vals.size();
        }

        double sum() {
            double sum = 0;
            for(Double val : this.vals) {
                sum += val;
            }
            return sum;
        }
    }

}
