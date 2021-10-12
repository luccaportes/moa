/*
 *    OzaBag.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package moa.classifiers.meta;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.core.*;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;
import com.github.javacliparser.IntOption;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Incremental on-line bagging of Oza and Russell.
 *
 * <p>Oza and Russell developed online versions of bagging and boosting for
 * Data Streams. They show how the process of sampling bootstrap replicates
 * from training data can be simulated in a data stream context. They observe
 * that the probability that any individual example will be chosen for a
 * replicate tends to a Poisson(1) distribution.</p>
 *
 * <p>[OR] N. Oza and S. Russell. Online bagging and boosting.
 * In Artiﬁcial Intelligence and Statistics 2001, pages 105–112.
 * Morgan Kaufmann, 2001.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classiﬁer to train</li>
 * <li>-s : The number of models in the bag</li> </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class DDCS extends AbstractClassifier implements MultiClassClassifier,
        CapabilitiesHandler {

    @Override
    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public IntOption chunkSizeOption = new IntOption("chunk", 'c',
            "The size of the chunks.", 1000, 1, Integer.MAX_VALUE);

    public IntOption nJobsOption = new IntOption("nJobs", 'w',
            "Number of threads to use", 1, 1, Integer.MAX_VALUE);

    public FlagOption initEnsembleOption = new FlagOption("initEnsemble", 'i',
        "Wheter to prestart the ensemble or not");

    public FlagOption baggingOption = new FlagOption("bagging", 'b',
        "Wheter to use online bagging when updating or not");

    public MultiChoiceOption votingMethodOption = new MultiChoiceOption("votingMethod", 'v',
            "",
            new String[]{"NO_SEL", "KNORAE", "KNORAU"},
            new String[]{"NO_SEL", "KNORAE", "KNORAU"},
            0);

    private ArrayList<Classifier> ensemble;
    private KDTree searcher;
    private ArrayList<BasicClassificationPerformanceEvaluator> evals;

    private Instances instanceBuffer;

    private int count = 0;

    @Override
    public void resetLearningImpl() {
        this.ensemble = new ArrayList<Classifier>();
        this.evals = new ArrayList<BasicClassificationPerformanceEvaluator>();
        if (this.initEnsembleOption.isSet()) {
            Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
            baseLearner.resetLearning();
            BasicClassificationPerformanceEvaluator ev = new BasicClassificationPerformanceEvaluator();
            for (int i = 0; i < this.ensembleSizeOption.getValue(); i++) {
                this.ensemble.add(baseLearner.copy());
                this.evals.add((BasicClassificationPerformanceEvaluator) ev.copy());
            }
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.instanceBuffer == null){
            this.instanceBuffer = new Instances(this.getModelContext(), this.chunkSizeOption.getValue());
        }
        if (this.instanceBuffer.size() <  this.chunkSizeOption.getValue()) {
            this.instanceBuffer.add(inst);
        } else {
            this.searcher = new KDTree();
            try {
                this.searcher.setInstances(this.instanceBuffer);
            } catch (Exception e) {
                e.printStackTrace();
            }
            Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
            BasicClassificationPerformanceEvaluator ev = new BasicClassificationPerformanceEvaluator();
            baseLearner.resetLearning();

            try {
                if (this.ensemble.size() == this.ensembleSizeOption.getValue()) {
                    int worst_index = get_worse();
                    this.ensemble.remove(worst_index);
                    this.evals.remove(worst_index);
                }
                this.ensemble.add(baseLearner.copy());
                this.evals.add((BasicClassificationPerformanceEvaluator) ev.copy());
            } catch (Exception e) {
                System.out.println("first time error");
            }

            ExecutorService pool = Executors.newFixedThreadPool(this.nJobsOption.getValue());
            if (this.baggingOption.isSet()) {
                for (int i = 0; i < this.ensemble.size(); i++) {
                    Runnable r = new trainerBag(this.ensemble.get(i), this.instanceBuffer, this.classifierRandom);
                    pool.execute(r);
                }
            } else {
                for (int i = 0; i < this.ensemble.size(); i++) {
                    Runnable r = new trainerNoBag(this.ensemble.get(i), this.instanceBuffer);
                    pool.execute(r);
                }
            }

            pool.shutdown();
            try {
                if (!pool.awaitTermination(60, TimeUnit.MINUTES)) {
                    pool.shutdownNow();
                }
            } catch (InterruptedException ex) {
                pool.shutdownNow();
                Thread.currentThread().interrupt();
            }
            this.instanceBuffer = null;
            this.instanceBuffer = new Instances(this.getModelContext(), this.chunkSizeOption.getValue());
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.votingMethodOption.getChosenLabel().equals("NO_SEL")){
            return getVotesForInstanceNoSel(inst);
        } else if (this.votingMethodOption.getChosenLabel().equals("KNORAE")){
            return getVotesForInstanceKNORAE(inst);
        } else if (this.votingMethodOption.getChosenLabel().equals("KNORAU")){
            return getVotesForInstanceKNORAU(inst);
        }
        return new double[0];
    }

    private double[] getVotesForInstanceKNORAU(Instance inst){
        Example testInst = new InstanceExample((Instance) inst);
        Instances neighbours;
        try {
            neighbours = this.searcher.kNearestNeighbours(inst, 7);
        } catch (Exception e) {
            return getVotesForInstanceNoSel(inst);
        }
        double[] result = new double[inst.numClasses()];

        for (int i = 0; i < this.ensemble.size(); ++i) {
            double[] votes = this.ensemble.get(i).getVotesForInstance(inst);
            DoubleVector voteVector = new DoubleVector(votes);
            int actualPredictedClass = voteVector.maxIndex();

            DoubleVector vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(inst));
            this.evals.get(i).addResult(testInst, vote.getArrayCopy());

            //int numVoteRep = 0;

            for (int j = 0; j < neighbours.size(); j++) {
                Instance neighbour = neighbours.get(j);
                int actualNeighClass = (int) neighbour.classValue();

                double[] votesNeigh = this.ensemble.get(i).getVotesForInstance(neighbour);
                DoubleVector voteVectorNeigh = new DoubleVector(votesNeigh);
                int predictedNeighClass = voteVectorNeigh.maxIndex();

                if (predictedNeighClass == actualNeighClass) {
                    result[actualPredictedClass]++;
                    //numVoteRep++;
                }
            }
        }

        return result;
    }


    private double[] getVotesForInstanceKNORAE(Instance inst){
        Example testInst = new InstanceExample((Instance) inst);
        Instances neighbours;
        try {
            neighbours = this.searcher.kNearestNeighbours(inst, 7);
        } catch (Exception e) {
            return getVotesForInstanceNoSel(inst);
        }
        double[] result = new double[inst.numClasses()];
        int[] correctVotesNumber = new int[this.ensemble.size()];

        for (int j = 0; j < this.ensemble.size(); ++j) {
            DoubleVector vote = new DoubleVector(this.ensemble.get(j).getVotesForInstance(inst));
            this.evals.get(j).addResult(testInst, vote.getArrayCopy());
        }


        for (int i = 0; i < neighbours.size(); i++) {
            Instance neighbour = neighbours.get(i);
            int actualClass = (int) neighbour.classValue();
            for (int j = 0; j < this.ensemble.size(); ++j) {

                double[] votes = this.ensemble.get(j).getVotesForInstance(neighbour);
                DoubleVector voteVector = new DoubleVector(votes);
                int predictedClass = voteVector.maxIndex();
                if (predictedClass == actualClass) {
                    correctVotesNumber[j] += 1;
                }
            }
        }

        int max = Arrays.stream(correctVotesNumber).max().getAsInt();
        int[] indexes = IntStream.range(0, correctVotesNumber.length).filter(i -> correctVotesNumber[i] == max).toArray();


        for (int i: indexes) {
            double[] votes = this.ensemble.get(i).getVotesForInstance(inst);
            DoubleVector voteVector = new DoubleVector(votes);
            int predictedClass = voteVector.maxIndex();
            if (predictedClass != -1){
                result[predictedClass] += 1;
            }
        }

        return result;
    }

    private double[] getVotesForInstanceNoSel(Instance inst) {
        Example testInst = new InstanceExample((Instance) inst);
        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.size(); i++) {
            DoubleVector vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(inst));
            this.evals.get(i).addResult(testInst, vote.getArrayCopy());
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                this.ensemble != null ? this.ensemble.size() : 0)};
    }

    @Override
    public Classifier[] getSubClassifiers() {
        Classifier[] arr = this.ensemble.toArray(new Classifier[this.ensemble.size()]);
        return arr.clone();
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == DDCS.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

    private int get_worse(){
        int index = 0;
        double worse = 101.0;
        for (int i = 0; i < this.evals.size(); i++){
            double curr_acc = this.evals.get(i).getPerformanceMeasurements()[1].getValue();
            if (curr_acc < worse){
                worse = curr_acc;
                index = i;
            }
        }
        return index;
    }
}

class trainerNoBag implements Runnable {
    private Classifier clf;
    private Instances insts;
    public trainerNoBag(Classifier clf, Instances insts){
        this.clf = clf;
        this.insts = insts;
    }

    @Override
    public void run() {
        for (int i=0; i< this.insts.size(); i++) {
            this.clf.trainOnInstance(this.insts.get(i));
        }
    }
}

class trainerBag implements Runnable {
    private Classifier clf;
    private Instances insts;
    private Random classifierRandom;
    public trainerBag(Classifier clf, Instances insts, Random classifierRandom){
        this.clf = clf;
        this.insts = insts;
        this.classifierRandom = classifierRandom;
    }

    @Override
    public void run() {
        for (int i=0; i< this.insts.size(); i++) {
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) insts.get(i).copy();
                weightedInst.setWeight(insts.get(i).weight() * k);
                this.clf.trainOnInstance(weightedInst);
            }
        }
    }
}