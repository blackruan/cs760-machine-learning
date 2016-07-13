import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class TAN 
{
	/** All the counts for attributes. */
	private double [][][] m_Counts;
	
	/** All the counts for twin attributes. */
	private double [][][][][] t_Counts;
	
	/** The counts of the classes. */
	private double [] m_classes;
	
	/** Conditional mutual informations. */
	private double [][] condMutInfo;
	
	/** Structure of TAN */
	private List<TANNode> tan;
	
	/**
	 * Trains the classifier with the provided training data
	 */
	public void train(Instances instances) 
	{
		m_Counts = new double[instances.numClasses()]
				[instances.numAttributes() - 1][0];
		t_Counts = new double[instances.numClasses()]
				[instances.numAttributes() - 1][0][0][0];
		m_classes = new double[instances.numClasses()];
		condMutInfo = new double[instances.numAttributes() - 1]
				[instances.numAttributes() - 1];
		tan = new ArrayList<TANNode>();
	    
		int i_AttIndex = 0;
		int j_AttIndex = 0;
	    Enumeration<?> i_EnumAtt = instances.enumerateAttributes();
	    while (i_EnumAtt.hasMoreElements()) 
	    {
	    	Attribute i_Attribute = (Attribute) i_EnumAtt.nextElement();
	    	for (int i = 0; i < instances.numClasses(); i++) 
	    	{
	    		m_Counts[i][i_AttIndex] = new double[i_Attribute.numValues()];
	    		t_Counts[i][i_AttIndex] = new double[i_Attribute.numValues()][0][0];
	    		for (int j = 0; j < i_Attribute.numValues(); j++)
	    		{
	    			t_Counts[i][i_AttIndex][j] = 
	    					new double[instances.numAttributes() - 1][0];
	    			j_AttIndex = 0;
		    	    Enumeration<?> j_EnumAtt = instances.enumerateAttributes();
		    	    while (j_EnumAtt.hasMoreElements()) 
		    	    {
		    	    	Attribute j_Attribute = (Attribute) j_EnumAtt.nextElement();
			    		t_Counts[i][i_AttIndex][j][j_AttIndex] = 
			    				new double[j_Attribute.numValues()];
			    		j_AttIndex++;
		    	    }
	    		}
	    	}
	    	i_AttIndex++;
	    }
	    
	    Enumeration<?> enumIns = instances.enumerateInstances();
	    while (enumIns.hasMoreElements()) 
	    {
	    	Instance instance = (Instance) enumIns.nextElement();
	    	i_AttIndex = 0;
	    	while (i_AttIndex < instances.numAttributes() - 1) 
	    	{
	    		m_Counts[(int)instance.classValue()][i_AttIndex]
	    				[(int)instance.value(i_AttIndex)]++; 
	    		j_AttIndex = 0;
	    		while (j_AttIndex < instances.numAttributes() - 1) 
		    	{
	    			t_Counts[(int)instance.classValue()][i_AttIndex]
	    					[(int)instance.value(i_AttIndex)][j_AttIndex]
		    						[(int)instance.value(j_AttIndex)]++; 
	    			j_AttIndex++;
		    	}
	    		i_AttIndex++;
	    	}
	    	m_classes[(int)instance.classValue()]++;
	    }
	    
	    for (i_AttIndex = 0; i_AttIndex < instances.numAttributes() - 1; i_AttIndex++)
		{
			for (j_AttIndex = 0; j_AttIndex < instances.numAttributes() - 1; j_AttIndex++)
			{
			    condMutInfo[i_AttIndex][j_AttIndex] = 
			    		calculateCMI(i_AttIndex, j_AttIndex, instances.numInstances());
			}
		}
//	    for(int i = 0; i < condMutInfo.length; i++)
//	    {
//		    for(int j = 0; j < condMutInfo[i].length; j++)
//		    {
//		    	System.out.printf(condMutInfo[i][j] + " ");
//		    }
//	    	System.out.println();
//	    }
	    Prim mst = new Prim();
	    mst.primAlg(condMutInfo, 0);
//	    for(int i = 0; i < mst.adjMatrix.length; i++)
//	    {
//	    	for(int j = 0; j < mst.adjMatrix.length; j++)
//	    	{
//	    		if(mst.adjMatrix[i][j] == true)
//	    		{
//	    			System.out.println("(" + i + ", " + j +")");
//	    		}
//	    	}
//	    }
	    for(int j = 0; j < instances.numAttributes() - 1; j++)
	    {
	    	List<Integer> parentIndexes = new ArrayList<Integer>();
	    	for(int i = 0; i < instances.numAttributes() - 1; i++)
	    	{
	    		if(mst.adjMatrix[i][j] == true)
	    		{
	    			parentIndexes.add(i);
	    		}
	    	}
	    	parentIndexes.add(instances.classIndex());
	    	List<Attribute> parents = new ArrayList<Attribute>();
	    	for (Integer i : parentIndexes)
	    	{
	    		parents.add(instances.attribute(i));
	    	}
	    	tan.add(new TANNode(instances.attribute(j), j, parents, parentIndexes));
	    }
	    
//    	List<Integer> parentIndexes = new ArrayList<Integer>();
//    	List<Attribute> parents = new ArrayList<Attribute>();
    	tan.add(new TANNode(instances.classAttribute(), instances.classIndex(),
    			null, null));
	    
	    for (Instance instance : instances)
	    {
	    	for (TANNode tanNode : tan)
	    	{
	    		tanNode.train(instance);
	    	}
	    }
	    
//	    for(int i = 0; i < 2; i++)
//	    {
//	    	for (int j = 0; j < 8; j++)
//	    	{
//	    		for (int k = 0; k < 2; k++)
//	    		{
//			    	List<Integer> list = new ArrayList<Integer>();
//		    		list.add(j);
//		    		list.add(k);
//		    		System.out.println(tan.get(3).getCondProb(i, list));
//	    		}
//	    	}
//	    }
	    
//	    for(int i = 0; i < 2; i++)
//	    {
//	    	List<Integer> list = new ArrayList<Integer>();
//		    System.out.println(tan.get(18).getCondProb(i, list));
//	    }
	}
	
	public ClassifyResult classify(Instance instance)
	{
		ClassifyResult result = new ClassifyResult();
	    double[] probs = new double[instance.numClasses()];
	    double bestProb = 0;
	    int bestClassIndex = 0;
	    
	    for (int i = 0; i < instance.numClasses(); i++) 
	    {
	    	probs[i] = 1;
	    	for (TANNode tanNode : tan)
	    	{
	    		probs[i] *= tanNode.getCondProb(instance, i);
	    	}
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
		for (int i = 0; i < tan.size() - 1; i++)
		{
			StringBuilder sb = new StringBuilder();
			TANNode tanNode = tan.get(i);
			sb.append(tanNode.getAttribute().name() + " ");
			List<Attribute> parents = tanNode.getParent();
			for (Attribute att : parents)
			{
				sb.append(att.name() + " ");
			}
			System.out.println(sb.toString());
		}
	}
	
	private double calculateCMI(int i_AttIndex, int j_AttIndex, int numIns)
	{
		if(i_AttIndex == j_AttIndex)
		{
			return -1.0;
		}
		else
		{
			double sum = 0;
			int numClasses = m_classes.length;
			for(int k = 0; k < numClasses; k++)
			{
				int iAtt_numValues = t_Counts[k][i_AttIndex].length;
				for(int i = 0; i < iAtt_numValues; i++)
				{
					int jAtt_numValues = 
							t_Counts[k][i_AttIndex][i][j_AttIndex].length;
					for(int j = 0; j < jAtt_numValues; j++)
					{
						double p_xi_xj_y = (t_Counts[k][i_AttIndex][i]
								[j_AttIndex][j] + 1) / 
								((double)numIns + 
								(double)(numClasses * iAtt_numValues *
								jAtt_numValues));
						double cp_xi_xj_y = (t_Counts[k][i_AttIndex][i]
								[j_AttIndex][j] + 1) / 
								(m_classes[k] + (double)(iAtt_numValues *
								jAtt_numValues));
						double cp_xi_y = (m_Counts[k][i_AttIndex][i] + 1) 
		    					/ (m_classes[k] + (double)iAtt_numValues);
						double cp_xj_y = (m_Counts[k][j_AttIndex][j] + 1) 
		    					/ (m_classes[k] + (double)jAtt_numValues);
						sum += p_xi_xj_y * Math.log(cp_xi_xj_y /
								(cp_xi_y * cp_xj_y)) / Math.log(2);
					}
				}
			}
			return sum;
		}
	}
}








