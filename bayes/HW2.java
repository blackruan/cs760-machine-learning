import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This problem should use weka.jar as external library. The package version I 
 * used is weka-3.7.3.jar
 */

public class HW2 
{
	public static void main(String[] args) throws Exception
	{
		if (args.length != 3) 
		{
			 System.out.println("usage: bayes <train-set-file> <test-set-file> <n|t>");
			 System.exit(-1);
		}
		DataSource train_set_file = new DataSource(args[0]);
		Instances train_set = train_set_file.getDataSet();
		if (train_set.classIndex() == -1)
		{
			train_set.setClassIndex(train_set.numAttributes() - 1);
		}
		DataSource test_set_file = new DataSource(args[1]);
		Instances test_set = test_set_file.getDataSet();
		if (test_set.classIndex() == -1)
		{
			test_set.setClassIndex(test_set.numAttributes() - 1);
		}
		
		String option = args[2];
		if (option.equals("n"))
		{
			int correctCount = 0;
			NaiveBayesClassifier nb = new NaiveBayesClassifier();
			nb.train(train_set);
			nb.print();
			System.out.println();
			for (Instance instance : test_set)
			{
				String actual = instance.stringValue(instance.classIndex());
				ClassifyResult result = nb.classify(instance);
				System.out.println(result.label + " " + actual + " " + result.prob);
				if (result.label.equals(actual))
				{
					correctCount++;
				}
			}
			System.out.println("\n" + correctCount);
		}
		else if (option.equals("t"))
		{
			int correctCount = 0;
			TAN tan = new TAN();
			tan.train(train_set);
			tan.print();
			System.out.println();
			for (Instance instance : test_set)
			{
				String actual = instance.stringValue(instance.classIndex());
				ClassifyResult result = tan.classify(instance);
				System.out.println(result.label + " " + actual + " " + result.prob);
				if (result.label.equals(actual))
				{
					correctCount++;
				}
			}
			System.out.println("\n" + correctCount);
		}
		else
		{
			System.out.println("usage: bayes <train-set-file> <test-set-file> <n|t>");
		}
	}
}
