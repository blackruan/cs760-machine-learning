import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This problem should use weka.jar as external library. The package version I used
 * is weka-3.7.3.jar
 */

public class HW1
{
	public static void main(String[] args) throws Exception
	{
		if (args.length != 3) 
		{
			 System.out.println("usage: dt-learn <train-set-file> <test-set-file> m");
			 System.exit(-1);
		 }
		DataSource train_set_file = new DataSource(args[0]);
		Instances train_set = train_set_file.getDataSet();
		if (train_set.classIndex() == -1)
			train_set.setClassIndex(train_set.numAttributes() - 1);
		int m = Integer.parseInt(args[2]);
		if (m > train_set.size())
		{
			System.out.println("m should be <= size of the train set");
			 System.exit(-1);
		}
		DecisionTree decTree = new DecisionTree(train_set, m);
		decTree.print();
		System.out.println();
		System.out.println("predicted class label    actual class label");
		DataSource test_set_file = new DataSource(args[1]);
		Instances test_set = test_set_file.getDataSet();
		if (test_set.classIndex() == -1)
			test_set.setClassIndex(test_set.numAttributes() - 1);
		int numberOfCorrect = 0;
		for (Instance test_instance : test_set)
		{
			String predicted = decTree.classify(test_instance);
			String actual = test_instance.stringValue(test_set.classAttribute());
			System.out.print(predicted);
			System.out.println("    " + actual);
			if (predicted.equals(actual))
			{
				numberOfCorrect ++;
			}
		}
		System.out.println(numberOfCorrect + "    " + test_set.size());
	}
}
