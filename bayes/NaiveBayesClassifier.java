import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.Utils;

public class NaiveBayesClassifier 
{
	/** All the counts for attributes. */
	private double [][][] m_Counts;
	
	/** The prior probabilities of the classes. */
	private double [] m_Priors;
	
	/** List of attributes of training examples */
	private List<Attribute> attributes;
	
	/**
	 * Trains the classifier with the provided training data
	 */
	public void train(Instances instances) 
	{
		m_Counts = new double[instances.numClasses()]
				[instances.numAttributes() - 1][0];
		m_Priors = new double[instances.numClasses()];
		attributes = new ArrayList<Attribute>();
		
		int attIndex = 0;
	    double sum;
	    
	    Enumeration<?> enumAtt = instances.enumerateAttributes();
	    while (enumAtt.hasMoreElements()) 
	    {
	    	Attribute attribute = (Attribute) enumAtt.nextElement();
	    	for (int i = 0; i < instances.numClasses(); i++) 
	    	{
	    		m_Counts[i][attIndex] = new double[attribute.numValues()];
	    	}
	    	attIndex++;
	    	attributes.add(attribute);
	    }
	    attributes.add(instances.classAttribute());
	    
	    Enumeration<?> enumIns = instances.enumerateInstances();
	    while (enumIns.hasMoreElements()) 
	    {
	    	Instance instance = (Instance) enumIns.nextElement();
	    	attIndex = 0;
	    	while (attIndex < instances.numAttributes() - 1) 
	    	{
	    		m_Counts[(int)instance.classValue()][attIndex]
	    				[(int)instance.value(attIndex)]++;
	    		attIndex++;
	    	}
	    	m_Priors[(int)instance.classValue()]++;
	    }
	    
	    // Normalize counts
	    enumAtt = instances.enumerateAttributes();
	    attIndex = 0;
	    while (enumAtt.hasMoreElements()) 
	    {
	    	Attribute attribute = (Attribute) enumAtt.nextElement();
	    	for (int i = 0; i < instances.numClasses(); i++) 
	    	{
	    		sum = Utils.sum(m_Counts[i][attIndex]);
	    		for (int j = 0; j < attribute.numValues(); j++) 
	    		{
	    			m_Counts[i][attIndex][j] = (m_Counts[i][attIndex][j] + 1) 
	    					/ (sum + (double)attribute.numValues());
	    		}
	    	}
	    	attIndex++;
	    }
	    
	    // Normalize priors
	    sum = Utils.sum(m_Priors);
	    for (int i = 0; i < instances.numClasses(); i++)
	    {
	    	m_Priors[i] = (m_Priors[i] + 1) / (sum + (double)instances.numClasses());
	    }
	}
	
	public ClassifyResult classify(Instance instance)
	{
		ClassifyResult result = new ClassifyResult();
	    double[] probs = new double[instance.numClasses()];
	    double bestProb = 0;
	    int bestClassIndex = 0;
	    int attIndex;
	    
	    for (int i = 0; i < instance.numClasses(); i++) 
	    {
	    	probs[i] = 1;
	    	attIndex = 0;
	    	while (attIndex < instance.numAttributes() - 1)
	    	{
	    		probs[i] *= m_Counts[i][attIndex][(int)instance.value(attIndex)];
	    		attIndex++;
	    	}
	    	probs[i] *= m_Priors[i];
	    	if (probs[i] > bestProb)
	    	{
	    		bestProb = probs[i];
	    		bestClassIndex = i;	    	
	    	}
	    }
	    
	 // Normalize probabilities
	    Utils.normalize(probs);
	    
	    result.label = instance.classAttribute().value(bestClassIndex);
	    result.prob = probs[bestClassIndex];
	    
	    return result;
	}
	
	public void print()
	{
		String className = attributes.get(attributes.size() - 1).name();
		for (int i = 0; i < attributes.size() - 1; i++)
		{
			System.out.println(attributes.get(i).name() + " " + className + " ");
		}
	}
}
