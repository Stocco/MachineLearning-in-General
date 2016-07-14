/**
 * Created by Stocco on 6/22/2016.
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.sql.Time;
import java.util.Random;
import java.util.Scanner;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;


import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import javax.swing.JPanel;

import javafx.scene.chart.ScatterChart;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.DomainOrder;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DatasetChangeListener;
import org.jfree.data.general.DatasetGroup;
import org.jfree.data.xy.*;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import javax.swing.JFrame;



import org.apache.commons.math3.linear.*;

import javax.swing.*;


public class LinearRegression {

    //training
    public  static double[][] train;
    public static double[] target;
    public static double[] betas;
    public static RealMatrix bet;
    public static XYSeries globSeries;

    // dev

    public static double[][] trainDev;
    public static double[] targetDev;


    public static void  main(String[] args)
    {
        loadTrainFiles(new File(args[0]), new File(args[1]));
        loadDevFiles(new File(args[2]), new File(args[3]));

        //Plotting chart
        /*
        XYSeries series = new XYSeries("Data Plot");
        XYSeries series2 = new XYSeries("Gradient");
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);
        dataset.addSeries(series2);
        JFreeChart chart = ChartFactory.createScatterPlot("scatter",null,null,dataset,PlotOrientation.VERTICAL,true,true,false);
        ChartFrame fr =  new ChartFrame("meu chart",chart);
        fr.setSize(500,500);
        fr.setVisible(true);
        globSeries = series2;

        for(int i=0;i<train.length;i++)
        {
            for(int j=1;j<train[i].length;j++)
            {
                System.out.println(i);
                series.add(train[i][j],target[i]);
            }
        }



        for(double z=dataset.getX(0,0).doubleValue();z<dataset.getX(0,train.length-1).doubleValue();z+=0.1)
        {
          series2.add(z,predict(new double[] {z}));
        }

         */


         gradientLinear();

//         Evaluate();

        System.out.print(bet);


    }

    static public void gradientLinear()
    {
        RealMatrix tr = new Array2DRowRealMatrix(train);
        RealMatrix bethat = new Array2DRowRealMatrix(betas);
        RealMatrix tar = new Array2DRowRealMatrix(target);
        boolean converged=false;

        while(!converged)
        {

            //process gradient
            RealMatrix aux = bethat.copy();

            bethat = tr.transpose().multiply(tr.multiply(bethat).subtract(tar));
            bethat = bethat.scalarMultiply(2);
            bethat = bethat.scalarMultiply(1.0/tr.getRowDimension());


            //advancing one step -> Step Size(SS) == 0.250        | formula: beta = beta - SS * gradient;
            bethat  = aux.subtract(bethat.scalarMultiply(0.1));


            //checking conversion Diff < 0.00001
            converged = converged(aux,bethat);


            /*//plotting gradient point in chart
            Scanner scan = new Scanner(System.in);
            globSeries.add(bethat.getEntry(0,0),bethat.getEntry(1,0));
            String hue = scan.next();*/
        }


        bet=bethat.copy();
    }

    static public void Evaluate()
    {

    }

    static public void closedForm()
    {
        //Initial values
        RealMatrix tr = new Array2DRowRealMatrix(train);
        RealMatrix bethat = new Array2DRowRealMatrix(betas);
        RealMatrix tar = new Array2DRowRealMatrix(target);


        //Process
        RealMatrix den = tr.transpose().multiply(tr);
        bethat = (tr.transpose().multiply(tar));
        den = new LUDecomposition(den).getSolver().getInverse();
        bethat = den.multiply(bethat);
        bet=bethat;
        System.out.print(bethat);

    }

    static public double predict(double[] input)
    {
        if(input.length != bet.getRowDimension() - 1)
        {
            return -1.0;
        }
        double totalsum=0.0;
        totalsum = bet.getEntry(0,0);
        for(int i=0;i<input.length;i++)
        {
           totalsum = totalsum + input[i] * bet.getEntry(i+1,0);
        }
        System.out.println("prediction: " + totalsum);

        return totalsum;
    }


    static public boolean converged(RealMatrix old, RealMatrix neW)
    {
        double sumOld = old.getNorm();
        double sumNew = neW.getNorm();

        if(sumOld == 0) return false;

        sumOld = Math.abs((sumOld - sumNew) / sumOld);

        System.out.println(sumOld);

        if(sumOld < 0.000001)
        {
            return true;
        }
        return false;
    }


    static public void loadTrainFiles(File x, File y)
    {

        Scanner auxscan = null;
        Scanner scanTrain = null;
        Scanner scanTarget = null;
        try {
            scanTrain = new Scanner(x);
            auxscan = new Scanner(x);
            scanTarget = new Scanner(y);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }



        // Section to know and generate the dimensions of the array
        int rows=0;
        int z = 0;
        while(auxscan.hasNext())
        {
            rows++;

            if(rows==1) {
                String current = auxscan.nextLine();
                for (String line : current.split(" ")) {
                    z++;
                }
            }
            else
            {
                auxscan.nextLine();
            }
        }
        train = new double[rows][z+1];
        target = new double[rows];
        betas = new double[z+1];
        betas[0] = 0;
        // ================================================================



        // parsing train values
        int r=0;
        while (scanTrain.hasNext())
        {
            int c=1;
            train[r][0] = 1;
            for (String line: scanTrain.nextLine().split(" ")) {
                train[r][c] = Double.parseDouble(line);
                c++;
            }
            r++;

        }

        // parsig targe values
        int r2=0;
        while(scanTarget.hasNext())
        {
            target[r2] = Double.parseDouble(scanTarget.nextLine());
            r2++;
        }




    }


    static public void loadDevFiles(File x ,File y)
    {
        Scanner auxscan = null;
        Scanner scanTrain = null;
        Scanner scanTarget = null;
        try {
            scanTrain = new Scanner(x);
            auxscan = new Scanner(x);
            scanTarget = new Scanner(y);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        int rows=0;
        int z = 0;
        while(auxscan.hasNext())
        {
            rows++;

            if(rows==1) {
                String current = auxscan.nextLine();
                for (String line : current.split(" ")) {
                    z++;
                }
            }
            else
            {
                auxscan.nextLine();
            }
        }
        trainDev = new double[rows][z];
        targetDev = new double[rows];


        int r=0;
        while (scanTrain.hasNext())
        {
            int c=0;
            for (String line: scanTrain.nextLine().split(" ")) {
                trainDev[r][c] = Double.parseDouble(line);
                c++;
            }
            r++;

        }

        // parsig targe values
        int r2=0;
        while(scanTarget.hasNext())
        {
            targetDev[r2] = Double.parseDouble(scanTarget.nextLine());
            r2++;
        }

    }







}
