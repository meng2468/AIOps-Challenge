import java.util.*;
import java.lang.*;
import java.io.*;
class Main {
    
    static char cur_name;
    
    // Recursively print the arrangement for minimum cost of multiplication
    static void printBracketsMatrixChain(int i, int j, int brackets[][]){
        
        // you have a single matrix ( you cannot further reduce the problem, so print the matrix )
        if(i == j){
            System.out.print(cur_name);
            cur_name++;
        } else {
            System.out.print("(");
            
            // Reduce the problem into left sub-problem ( left of optimal arrangement )
            printBracketsMatrixChain(i, brackets[i][j], brackets);
            
            // Reduce the problem into right sub-problem ( right of optimal arrangement )
            printBracketsMatrixChain(brackets[i][j]+1, j, brackets);
            System.out.print(")");
        }
    }
    static void matrixMultiplicationProblem(int matrixSize[]) {
        int numberOfMatrices = matrixSize.length-1;
    
        // dp[i][j] = minimum number of operations required to multiply matrices i to j
        int dp[][] = new int[numberOfMatrices][numberOfMatrices];
    
        // initialising dp array with Integer.MAX_VALUE ( maximum number of operations )
        for(int i=0;i<numberOfMatrices;i++){
            for(int j=0;j<numberOfMatrices;j++){
                dp[i][j] = Integer.MAX_VALUE;
                if(i == j) // for a single matrix from i to i, cost = 0
                    dp[i][j] = 0;
            }
        }
    
        int brackets[][] = new int[numberOfMatrices][numberOfMatrices];
        for(int len=2;len<=numberOfMatrices;len++){
            for(int i=0;i<numberOfMatrices-len+1;i++){
                int j = i+len-1;
                for(int k=i;k<j;k++) {
                    int val = dp[i][k]+dp[k+1][j]+(matrixSize[i]*matrixSize[k+1]*matrixSize[j+1]);
                    if(val < dp[i][j]) {
                        dp[i][j] = val;
                        brackets[i][j] = k;
                    }
                }
            }
        }
    
        // naming the first matrix as A
        cur_name = 'A';
        
        // calling function to print brackets
        printBracketsMatrixChain(0, numberOfMatrices-1, brackets);
        System.out.println();
    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        while(t-- > 0) {
            int n = sc.nextInt();
            int matrixSize[] = new int[n];
            for(int i=0;i<n;i++) matrixSize[i] = sc.nextInt();
            matrixMultiplicationProblem(matrixSize);
        }
    }
}