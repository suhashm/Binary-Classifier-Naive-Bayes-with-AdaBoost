package classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;

public class NaiveBayes {
	
	public static int numOfAttributes;
	public static int numOfLines;
	public static int predictedClassForADABoost;
	public static ArrayList<Integer> errListIndexForADABoost; // for each index add 1 or 0 based on correct prediction, used to calculate Model error-ADABoost
	public static Hashtable<Integer, Hashtable<Integer, Integer>> trainAttrUniqVal; // store number of unique values for an attribute
	public static ArrayList<Integer> attributeList; // to put attr:0 for each row in train file
	public static Hashtable<Integer, Integer> trainClassListCount; // Keep count of number of +1 s and -1 s in training data
	public static ArrayList<Integer> trainClassValues; // value of +1 and -1s for each line indexed from line 0 to line n-1 for training Data
	public static ArrayList<Integer> testClassValues; // value of +1 and -1s for each line indexed from line 0 to line n-1 for test Data
	
	// find number of attributes present in both training and test file, this is later used to add <attr>:0 in tuple of training set
	public static void findNumberOfAttributes(File trainFile, File testFile) throws IOException{
		numOfAttributes = 0;
		numOfLines = 0;
		attributeList = new ArrayList<Integer>();
		trainAttrUniqVal = new Hashtable<Integer, Hashtable<Integer, Integer>>();
		BufferedReader br = new BufferedReader(new FileReader(trainFile));
		Hashtable<Integer, Integer> trainAttributes = new Hashtable<Integer, Integer>();
		Hashtable<Integer, Integer> testAttributes = new Hashtable<Integer, Integer>();
		String line;
		while((line = br.readLine()) != null){
			String[] words = line.split(" ");
			for(int i = 1; i < words.length; i++){
				String[] attrValues = words[i].split(":");
				trainAttributes.put(Integer.parseInt(attrValues[0]), 1);
			}
		}
		
		BufferedReader br2 = new BufferedReader(new FileReader(testFile));
		String line2;
		while((line2 = br2.readLine()) != null){
			String[] words = line2.split(" ");
			for(int i = 1; i < words.length; i++){
				String[] attrValues = words[i].split(":");
				testAttributes.put(Integer.parseInt(attrValues[0]), 1);
			}
		}
		
		numOfAttributes = trainAttributes.size() > testAttributes.size() ? trainAttributes.size():testAttributes.size();
		
		for(int j = 1; j <=numOfAttributes; j++){
			attributeList.add(j);
		}
	}

	// Step 1: Load data from Train and Test, Pass 1 for training data
	public static ArrayList<Hashtable<Integer, Integer>> loadFromGivenData(File file, int trainOrTest) throws IOException{
		
		if(trainOrTest == 1){ // passing 1 to load train data
			trainClassValues = new ArrayList<Integer>();
			trainClassListCount = new Hashtable<Integer, Integer>();
			
		}else{
			testClassValues = new ArrayList<Integer>();
		}
		ArrayList<Hashtable<Integer, Integer>> trainTupleValues = new ArrayList<Hashtable<Integer, Integer>>();
		
		
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		while((line = br.readLine()) != null){
			
			Hashtable<Integer, Integer> lineAttrValue = new Hashtable<Integer, Integer>();
			
			String[] words = line.split(" ");
			if(words[0].equals("")){
				break;
			}
			
			char c1 = words[0].charAt(0);
			int c2 = words[0].charAt(1) - '0';
			int vval = 0;
			if(c1 == '+')
				vval = c2;
			else 
				vval = Integer.parseInt(words[0]);
			if(trainOrTest == 1){
				if(c1 == '+')
					trainClassValues.add(c2);
				else
					trainClassValues.add(Integer.parseInt(words[0]));
				
				// get count of +1 and -1 only for training data
				if(trainClassListCount.get(vval) == null){
					trainClassListCount.put(vval, 1);
				}else{
					int val = trainClassListCount.get(vval);
					trainClassListCount.put(vval, val+1);
				}
			}else{
				testClassValues.add(vval);
			}
			
			for(int i = 1; i < words.length; i++){
				String[] tup = words[i].split(":");
				lineAttrValue.put(Integer.parseInt(tup[0]), Integer.parseInt(tup[1]));
				
				// get the count of number of unique value for an attribute - Laplace Transform
				if(trainOrTest == 1){
					if(trainAttrUniqVal.get(Integer.parseInt(tup[0])) != null){
						Hashtable<Integer, Integer> uniqVals = trainAttrUniqVal.get(Integer.parseInt(tup[0]));
						uniqVals.put(Integer.parseInt(tup[1]), 1);
						trainAttrUniqVal.put(Integer.parseInt(tup[0]), uniqVals);
					}
					else{
						Hashtable<Integer, Integer> uniqVals = new Hashtable<Integer, Integer>();
						uniqVals.put(Integer.parseInt(tup[1]), 1);
						trainAttrUniqVal.put(Integer.parseInt(tup[0]), uniqVals);
					}
				}
			}
			
			// add not present attribute as attr:0
			for(int j = 0; j<attributeList.size(); j++){
				if(lineAttrValue.get(attributeList.get(j)) == null){
					lineAttrValue.put(attributeList.get(j), 0);
					
					// get the count of number of unique value for an attribute - Laplace Transform
					if(trainOrTest == 1){
						if(trainAttrUniqVal.get(attributeList.get(j)) != null){
							Hashtable<Integer, Integer> uniqVals = trainAttrUniqVal.get(attributeList.get(j));
							uniqVals.put(0, 1);
							trainAttrUniqVal.put(attributeList.get(j), uniqVals);
						}
						else{
							Hashtable<Integer, Integer> uniqVals = new Hashtable<Integer, Integer>();
							uniqVals.put(0, 1);
							trainAttrUniqVal.put(attributeList.get(j), uniqVals);
						}
					}
				}
			}
			trainTupleValues.add(lineAttrValue);	
		}
		return trainTupleValues;
	}
	
	// build classifier for the provided data set
	public static Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>> buildClassifier(ArrayList<Hashtable<Integer, Integer>> trainingData){
		Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>> classifier = new Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>>();
		
		for(int i = 0; i < trainingData.size(); i++){
			Hashtable<Integer, Integer> lineDataList = trainingData.get(i);
			for(int attr: lineDataList.keySet()){
				if(classifier.get(attr) == null){
				// add attribute to the classifier
					int mainKey = attr;
					int mainVal = lineDataList.get(attr);
					int secondKey = trainClassValues.get(i);
					double secondVal = 1.0;
					Hashtable<Integer, Double> thirdTable = new Hashtable<Integer, Double>();
					thirdTable.put(secondKey, secondVal);
					Hashtable<Integer, Hashtable<Integer, Double>> secondTable = new Hashtable<Integer, Hashtable<Integer, Double>>();
					secondTable.put(mainVal, thirdTable);
					classifier.put(mainKey, secondTable);
				}else{
					Hashtable<Integer, Hashtable<Integer, Double>> firstHT = classifier.get(attr);
					int mainVal = lineDataList.get(attr);
					if(firstHT.get(mainVal) == null){
						int secondKey = mainVal;
						int thirdKey = trainClassValues.get(i);
						double thirdVal = 1.0;
						Hashtable<Integer, Double> thirdTable = new Hashtable<Integer, Double>();
						thirdTable.put(thirdKey, thirdVal);
						firstHT.put(secondKey, thirdTable);
						classifier.put(attr, firstHT);
					}else{
						Hashtable<Integer, Double> innerHT = firstHT.get(mainVal);
						if(innerHT.get(trainClassValues.get(i)) == null){
							innerHT.put(trainClassValues.get(i), 1.0);
							firstHT.put(mainVal, innerHT);
							classifier.put(attr, firstHT);
						}else{
							double newVal = innerHT.get(trainClassValues.get(i));
							innerHT.put(trainClassValues.get(i), newVal + 1);
							firstHT.put(mainVal, innerHT);
							classifier.put(attr, firstHT);
						}
					}
				}				
			}

		}
		return classifier;
	}
	
	// predict labels on test data using above classifier
	public static void predictLables(ArrayList<Hashtable<Integer, Integer>> testData, Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>> classifier, int trainOrTest, boolean shouldPrint){
		predictedClassForADABoost = 0;
		if(trainOrTest == 1){
			errListIndexForADABoost = new ArrayList<Integer>(); // initialize err list count for Model error calculation-ADABoost
		}
		
		int numCorrectPredict = 0;
		int numOfZeroRes = 0;
		int TP = 0, TN = 0, FP = 0, FN = 0;
		for(int j = 0; j < testData.size(); j++){ // for each of line in train/ test data
			double max = 0.0;
			int predictedClass = 1; // set default predicted class to +1
			for(int i : trainClassListCount.keySet()){ // for each +1 and -1 class
				Hashtable<Integer, Integer> tempTestList = testData.get(j);
				double res = 1.0;
				for(int k :tempTestList.keySet()){ // for each of the attribute in the line
					if(classifier.get(k).get(tempTestList.get(k)) != null){ // check whether value of attribute in test set is present in classifier
						if(classifier.get(k).get(tempTestList.get(k)).get(i) != null) // check if +1 or -1 is present for that attribute
							res *= ((classifier.get(k).get(tempTestList.get(k)).get(i))+1)/(((trainClassListCount.get(i)) + trainAttrUniqVal.get(k).size()) * 1.0);
						else
							res *= 1/((((trainClassListCount.get(i)) + trainAttrUniqVal.get(k).size()) * 1.0)); //Apply Laplacian transform for (+1 or -1 ) not present for an attribute
					}else{
						res *= 1/((((trainClassListCount.get(i)) + trainAttrUniqVal.get(k).size()) * 1.0)); // Apply Laplace transform if Value of an attribute is not present in classifier
					}
				}
				if(res == 0)
					numOfZeroRes++;
				int acc = trainClassListCount.get(i);
				double ans = 0.0;
				ans = res * (trainClassListCount.get(i)/(numOfLines * 1.0));
				if(ans > max){
					max = ans; 
					predictedClass = i;
				}
			}
			predictedClassForADABoost = predictedClass;
			
			//calculate accuracy for training and test set separately
			if(trainOrTest != 1){
				if(testClassValues.get(j) == predictedClass)
					numCorrectPredict++;
			}else{
				if(trainClassValues.get(j) == predictedClass)
					numCorrectPredict++;
			}
			
			// calculate TP,FN,TN and FP separately for train and test data set, trainOrTest = 1 for training data
			if(trainOrTest != 1){
				if(testClassValues.get(j) > 0){
					if(testClassValues.get(j) == predictedClass)
						TP++;
					else
						FN++;
				}else{
					if(testClassValues.get(j) == predictedClass)
						TN++;
					else
						FP++;
				}
			}else{
				if(trainClassValues.get(j) > 0){
					if(trainClassValues.get(j) == predictedClass){
						errListIndexForADABoost.add(0);
						TP++;
					}						
					else{
						errListIndexForADABoost.add(1);
						FN++;
					}
						
				}else{
					if(trainClassValues.get(j) == predictedClass){
						errListIndexForADABoost.add(0);
						TN++;
					}
					else{
						errListIndexForADABoost.add(1);
						FP++;
					}
				}
			}
		}
		if(shouldPrint){
			System.out.println(TP+" "+FN+" "+FP+" "+TN);
			calculateEvaluation(TP, FN, FP, TN, trainOrTest);
		}
	}
	public static double roundVal(double val){
		return (double)Math.round(val * 1000) / 1000;
	}
	public static void calculateEvaluation(int TP,int FN,int FP,int TN, int trainOrTest){
		int all = TP + FN + FP + TN;
		double accuracy = (TP + TN) / (all * 1.0);
		double errorRate = 1 - accuracy;
		int P = TP + FN;
		int N = FP + TN;
		double sensitivity = TP /(P * 1.0);
		double specificity = TN / (N * 1.0);
		double precision = TP / ((TP + FP) * 1.0);
		double recall = TP/ ((TP + FN) * 1.0);
		
		double f1Score = (2 * precision * recall)/ ((precision + recall) * 1.0);
		double fb5 = ((1 + Math.pow(0.5, 2)) * precision * recall)/ ((((Math.pow(0.5, 2)) * precision) + recall) * 1.0); 
		double fb2 = ((1 + Math.pow(2, 2)) * precision * recall)/ ((((Math.pow(2, 2)) * precision) + recall) * 1.0);
	}
	
	public static void main(String[] args) throws IOException {
		String[] files = {"custom.train", //0
							"custom.test",
							"led.train",//2
						  "led.test",
						  "breast_cancer.train", //4
						  "breast_cancer.test",
						  "poker.train", //6
						  "poker.test",
						  "adult.train", //8
						  "adult.test"
						};
		if(args.length < 2){
			System.out.println("Please provide proper command line arguments");
			System.out.println("Usage: arg0: train file, arg1: test file");
			System.exit(0);
		}
		File trainFile = new File(args[0]);
		File testFile = new File(args[1]);
//		File trainFile = new File(files[2]);
//		File testFile = new File(files[3]);
		
		ArrayList<Hashtable<Integer, Integer>> trainingData = new ArrayList<Hashtable<Integer, Integer>>();
		ArrayList<Hashtable<Integer, Integer>> testData = new ArrayList<Hashtable<Integer, Integer>>();
		Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>> classifier = new Hashtable<Integer, Hashtable<Integer, Hashtable<Integer, Double>>>();
		findNumberOfAttributes(trainFile, testFile);
		
		// Step 1: load training data into memory, pass 1 as second parameter for train file
		trainingData = loadFromGivenData(trainFile, 1);		
		numOfLines = trainingData.size();
	
		// Step 1: load test data into memory, pass 0 as second parameter for test file
		testData = loadFromGivenData(testFile, 0);
		
		// Step 2: Build NaiveBayes Classifier
		classifier = buildClassifier(trainingData);
		
		// Step 2: use test data to predict labels using above classifier, pass 1 as last argument for running classifier on trainData
		predictLables(trainingData, classifier, 1, true);
		predictLables(testData, classifier, 0, true);
	}

}
