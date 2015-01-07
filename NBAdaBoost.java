package classification;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;

import classification.NaiveBayes;


public class NBAdaBoost {
	
	public static ArrayList<Integer> selectedTupleIndex; // maintain index of the randomly selected tuples
	
	// Apply initial weights for Adaboost i.e. 1/d
	public static ArrayList<Double> applyInitialWeight(ArrayList<Hashtable<Integer, Integer>> trainingData){
		ArrayList<Double> initialWeights = new ArrayList<Double>();
		for(int i = 0; i < trainingData.size(); i++){
			initialWeights.add(1/(trainingData.size() * 1.0));
		}
		return initialWeights;
	}
	
	// get sample based on the weights of each tuple
	public static ArrayList<Hashtable<Integer, Integer>> getSampleBasedOnWeights(ArrayList<Hashtable<Integer, Integer>> trainingData, ArrayList<Double> initialWeights){
		selectedTupleIndex = new ArrayList<Integer>(); // reset selected tuple for each of sample method call
		
		ArrayList<Hashtable<Integer, Integer>> sampledData = new ArrayList<Hashtable<Integer, Integer>>();
		
		for(int i = 0; i < trainingData.size(); i++){
			double b = Math.random();
			double sum = 0.0;
			for(int j = 0; j < initialWeights.size(); j++){
				sum += initialWeights.get(j);
				if(sum > b){
					selectedTupleIndex.add(j);
					sampledData.add(trainingData.get(j));
					break;
				}
			}
		}			
		return sampledData;
	}
	
	// Apply AdaBoost method, which is an ensemble classifier for the provided data set.
	public static void applyEnsembleClassifier(int k, ArrayList<Hashtable<Integer, Integer>> testData, ArrayList<Double> errOfModels,Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>>> storedModel, int trainOrTest ){
		int correctPredictADA = 0;
		int TP = 0, TN = 0, FP = 0, FN = 0;
		for(int i = 0; i < testData.size(); i++){
			
			// Hashtable to store the weight of each model for corresponding class prediction
			Hashtable<Integer, Double> predictClass = new Hashtable<Integer, Double>();
			
			for(int j = 1; j <= k; j++){
				double modelError = errOfModels.get((j-1)); // for Model 1, err will be in 0th index
				double modelWeight = Math.log((1-modelError)/(modelError * 1.0));
				Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>> modelI = storedModel.get(j);
				
				ArrayList<Hashtable<Integer, Integer>> testTuple = new ArrayList<Hashtable<Integer, Integer>>();
				testTuple.add(testData.get(i));
				NaiveBayes.predictLables(testTuple, modelI, 0, false);
				int classPrediction = NaiveBayes.predictedClassForADABoost;
				
				// Add each class prediction with sum of the modelWeights, then get the maximum value class, which is the prediction required
				if(predictClass.get(classPrediction) == null){
					predictClass.put(classPrediction, modelWeight);
				}else{
					double currWeight = predictClass.get(classPrediction);
					predictClass.put(classPrediction, currWeight+modelWeight);
				}
			}
			
			// get the class with maximum weight, which is the required prediction from ADABoost
			double maxWeight = 0.0;
			int finalPrediction = 1; // default predicted class
			for(int className: predictClass.keySet()){
				if(predictClass.get(className) > maxWeight){
					maxWeight = predictClass.get(className);
					finalPrediction = className;
				}
			}
			
			// get the count of correct predictions separate for train and test data set
			if(trainOrTest != 1){
				if(finalPrediction == NaiveBayes.testClassValues.get(i))
					correctPredictADA++;
			}else{
				if(finalPrediction == NaiveBayes.trainClassValues.get(i))
					correctPredictADA++;
			}
			
			// get count of TP, FN, FP, TN separately for test and train data set
			if(trainOrTest != 1){
			if(NaiveBayes.testClassValues.get(i) > 0){
					if(NaiveBayes.testClassValues.get(i) == finalPrediction)
						TP++;
					else
						FN++;
				}else{
					if(NaiveBayes.testClassValues.get(i) == finalPrediction)
						TN++;
					else
						FP++;
				}
			}else{
				if(NaiveBayes.trainClassValues.get(i) > 0){
					if(NaiveBayes.trainClassValues.get(i) == finalPrediction)
						TP++;
					else
						FN++;
				}else{
					if(NaiveBayes.trainClassValues.get(i) == finalPrediction)
						TN++;
					else
						FP++;
				}
			}
			
		}
	
		System.out.println(TP+" "+FN+" "+FP+" "+TN);
		NaiveBayes.calculateEvaluation(TP, FN, FP, TN, 0);
	}
	
	// calculate the sampled class values count for predictLabels
	public static Hashtable<Integer,Integer> getSampledClassValueCount(ArrayList<Integer> sampledTrainClassValues){
		Hashtable<Integer,Integer> classValCount = new Hashtable<Integer,Integer>();
		for(int i = 0; i < sampledTrainClassValues.size(); i++){
			if(classValCount.get((sampledTrainClassValues.get(i))) != null){
				int val = classValCount.get((sampledTrainClassValues.get(i)));
				classValCount.put((sampledTrainClassValues.get(i)),val+1);
			}else{
				classValCount.put((sampledTrainClassValues.get(i)),1);
			}
		}
		return classValCount;
	}
	
	// calculate number of unique attribute values in sampled data
	public static Hashtable<Integer, Hashtable<Integer, Integer>> getSampleTrainAttrUniqVal(ArrayList<Hashtable<Integer, Integer>> sampledTrainData){
		Hashtable<Integer, Hashtable<Integer, Integer>> resultTrainUniqAttr = new Hashtable<Integer, Hashtable<Integer, Integer>>();
		for(int i = 0; i < sampledTrainData.size(); i++){
			Hashtable<Integer, Integer> interimSample = new Hashtable<Integer, Integer>();
			interimSample = sampledTrainData.get(i);
			for(int k : interimSample.keySet()){
				if(resultTrainUniqAttr.get(k) != null){
					Hashtable<Integer, Integer> tempHT = resultTrainUniqAttr.get(k);
					tempHT.put(interimSample.get(k), 1);
					resultTrainUniqAttr.put(k, tempHT);
				}else{
					Hashtable<Integer, Integer> tempHT = new Hashtable<Integer, Integer>();
					tempHT.put(interimSample.get(k), 1);
					resultTrainUniqAttr.put(k, tempHT);
				}
			}
		}
		return resultTrainUniqAttr;
		
	}
	
	
	public static void main(String[] args) throws IOException {
		String[] files = {
			  "custom.train",        //0
			  "custom.test",
			  "led.train",           //2
			  "led.test",
			  "breast_cancer.train", //4
			  "breast_cancer.test",
			  "poker.train",         //6
			  "poker.test",
			  "adult.train", 		 //8
			  "adult.test"
			};
//		File trainFile = new File(files[6]);
//		File testFile = new File(files[7]);
		
		if(args.length < 2){
			System.out.println("Please provide proper command line arguments");
			System.out.println("Usage: arg0: train file, arg1: test file");
			System.exit(0);
		}
		File trainFile = new File(args[0]);
		File testFile = new File(args[1]);
		
		ArrayList<Hashtable<Integer, Integer>> trainingData = new ArrayList<Hashtable<Integer, Integer>>();
		ArrayList<Hashtable<Integer, Integer>> testData = new ArrayList<Hashtable<Integer, Integer>>();
		NaiveBayes.findNumberOfAttributes(trainFile, testFile);
		
		// load training data into memory, pass 1 as second parameter for train file
		trainingData = NaiveBayes.loadFromGivenData(trainFile, 1);
		//System.out.println("size of training data is "+trainingData.size());
		ArrayList<Integer> refTrainClassValues = new ArrayList<Integer>();		
		refTrainClassValues = (ArrayList<Integer>) NaiveBayes.trainClassValues.clone();
		
		Hashtable<Integer, Integer> refTrainClassListCount = new Hashtable<Integer, Integer>();
		refTrainClassListCount = (Hashtable<Integer, Integer>) NaiveBayes.trainClassListCount.clone();
		
		Hashtable<Integer, Hashtable<Integer, Integer>> refTrainAttrUniqVal = new Hashtable<Integer, Hashtable<Integer, Integer>>();
		refTrainAttrUniqVal = (Hashtable<Integer, Hashtable<Integer, Integer>>) NaiveBayes.trainAttrUniqVal.clone();
		
		testData = NaiveBayes.loadFromGivenData(testFile, 0);
				
		// apply inital weights to each tuple i.e. w1 =  1/d
		ArrayList<Double> initialWeights = applyInitialWeight(trainingData);
		ArrayList<Double> backupInitialWeights = new ArrayList<Double>();
		backupInitialWeights = (ArrayList<Double>) initialWeights.clone();
		// Apply adaboost for K classifiers
		int  k = 5; // number of classifiers
		double err = 0.0;
		
		// Hashtable to store index of classifier, classifier(model)  for predicting class label using ADABoost
		Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>>> storedModel = new Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>>>();
		
		// Store error of each model Mi for ADABoost calculation
		ArrayList<Double> errOfModels = new ArrayList<Double>();
		boolean errFlag = false;
		int errCount = 0;
		
		// Apply ADABoost for K Classifiers
		for( int i = 1; i <= k; i++){
			ArrayList<Hashtable<Integer, Integer>> sampledTrainingData = new ArrayList<Hashtable<Integer, Integer>>();
			Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>> classifier = new Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>>();
			double fsum = 0.0;
			for(int jb = 0; jb < initialWeights.size(); jb++)
				fsum += initialWeights.get(jb);
			
			if(errCount > 10){
				initialWeights = (ArrayList<Double>) backupInitialWeights.clone();
				errCount = 0;
			}
			sampledTrainingData = getSampleBasedOnWeights(trainingData, initialWeights); // get sampled data based on probability of weights
		
			// Before calling buildClassifier on sampled Data, set the trainClassValues to new class values as per sample data
			ArrayList<Integer> sampledTrainClassValues = new ArrayList<Integer>();
			for(int st = 0; st < sampledTrainingData.size(); st++){
				int nval = NaiveBayes.trainClassValues.get(selectedTupleIndex.get(st));
				sampledTrainClassValues.add(nval);
			}
			
			NaiveBayes.trainClassValues = (ArrayList<Integer>) sampledTrainClassValues.clone();
			classifier = NaiveBayes.buildClassifier(sampledTrainingData); // build model based on sampled Training data
			
			//count number of +1 and -1 - trainClassListCount
			Hashtable<Integer,Integer> sampleClassCount = getSampledClassValueCount(sampledTrainClassValues);
			
			// get sample Attrs unique value count
			Hashtable<Integer, Hashtable<Integer, Integer>> sampleTrainAttrUniqVal = new Hashtable<Integer, Hashtable<Integer, Integer>>();
			sampleTrainAttrUniqVal = getSampleTrainAttrUniqVal(sampledTrainingData);
			
			// Before calling predictLables on training Data, set the trainClassValues back to original value from sampled train class values
			NaiveBayes.numOfLines = trainingData.size();
			NaiveBayes.trainClassValues = (ArrayList<Integer>) refTrainClassValues.clone();
			
			// Before calling preditLables, set trainClassListCount and trainAttrUniqVal to sampled data			
			NaiveBayes.trainAttrUniqVal = (Hashtable<Integer, Hashtable<Integer, Integer>>) sampleTrainAttrUniqVal.clone();
			NaiveBayes.trainClassListCount = (Hashtable<Integer, Integer>) sampleClassCount.clone();
			
			NaiveBayes.predictLables(trainingData, classifier, 1, false);
			
			// calculate error of model
			err = 0.0;
			for(int j = 0; j < sampledTrainingData.size(); j++){
				double intw = initialWeights.get(j);
				int erlist = NaiveBayes.errListIndexForADABoost.get(j);
				err = err + (intw * erlist);
			}
			
			if(err < 0.5){
				errCount = 0;
				
				// Add index of classifier, classifier(model) and error in Hashtable for predicting class label using ADABoost
				storedModel.put(i, classifier);
				errOfModels.add(err); // refer to i-1 index i.e. for model 1 err is in errOfModels(0)
				errFlag = false;
				
				// update the weight of correctly classified tuples
				for(int p = 0; p < NaiveBayes.errListIndexForADABoost.size(); p++){
					if(NaiveBayes.errListIndexForADABoost.get(p) == 0){ // correctly classified, refer predictLabels method of NaiveBayes
						double newWeight = initialWeights.get(selectedTupleIndex.get(p)) * (err/ ((1-err) * 1.0));
						initialWeights.set(selectedTupleIndex.get(p), newWeight);
					}
				}
				
				// Normalize the weights for all tuples in training set
				double newSum = 0.0;
				for(int q = 0; q < initialWeights.size(); q++){
					newSum += initialWeights.get(q);
				}
				// set new weights to all the tuples in training set
				for(int q1 = 0; q1 < initialWeights.size(); q1++){
					initialWeights.set(q1, (initialWeights.get(q1) / (newSum * 1.0)));
				}
				// test if new weight add is successfull by checking the sum, which should be = 1
				double testSum = 0.0;
				for(int q11 = 0; q11 < initialWeights.size(); q11++){
					testSum += initialWeights.get(q11);
				}
			
			}else{			
				errCount++;
				i-- ; // handle case where error is more than 0.5
				
			}
		}
		
		// Apply ensemble method to classify Tuple X of train and test data separately
		NaiveBayes.trainClassValues = (ArrayList<Integer>) refTrainClassValues.clone();
		NaiveBayes.trainAttrUniqVal = (Hashtable<Integer, Hashtable<Integer, Integer>>) refTrainAttrUniqVal.clone();
		NaiveBayes.trainClassListCount = (Hashtable<Integer, Integer>) refTrainClassListCount.clone();
		
		applyEnsembleClassifier(k, trainingData, errOfModels, storedModel, 1); // final parameter is 0 or 1 for test and training respectively.
		applyEnsembleClassifier(k, testData, errOfModels, storedModel, 0); // final parameter is 0 or 1 for test and training respectively.
	}
}
