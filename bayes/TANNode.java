import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Utils;


public class TANNode 
{
	/** Attribute of this TANNode. */
	private Attribute att;
	
	/** Attribute Index of this TANNode. */
	private int attIndex;
	
	/** Parent Attributes of this TANNode. */
	private List<Attribute> parents;
	
	/** Parent Attribute of this TANNode. */
	private List<Integer> parentIndexes;
	
	/** All the counts for this TANNode. */
	public double[] counts;
	
	/** Used to locate the number in counts */
	private int[] digits;
	
	public TANNode(Attribute _att, int _attIndex, List<Attribute> _parents, 
			List<Integer> _parentIndexes)
	{
		att = _att;
		attIndex = _attIndex;
		parents = _parents;
		parentIndexes = _parentIndexes;
		if (_parents == null || _parents.isEmpty())
		{
			digits = new int[1];
			digits[0] = 1;
			counts = new double[_att.numValues()];
		}
		else
		{
			digits = new int[_parents.size() + 1];
			int sum = 1;
			digits[digits.length - 1] = sum;
			for (int i = _parents.size() - 1; i >= 0; i--)
			{
				Attribute attribute = _parents.get(i);
				sum *= attribute.numValues();
				digits[i] = sum;
			}
			sum *= _att.numValues();
			counts = new double[sum];
		}
	}
	

	public Attribute getAttribute()
	{
		return att;
	}
	
	public List<Attribute> getParent()
	{
		return parents;
	}
	
	public void train(Instance instance)
	{
		if (parents == null || parents.isEmpty())
		{
			int valIndex = (int) instance.value(attIndex);
			counts[valIndex] ++;
		}
		else
		{
			int valIndex = (int) instance.value(attIndex);
			List<Integer> parentValIndex = new ArrayList<Integer>();
			for(Integer i : parentIndexes)
			{
				parentValIndex.add( (int) instance.value(i) );
			}
			counts[getPosition(valIndex, parentValIndex)] ++;
		}
	}
	
	public double getCondProb(int valIndex, List<Integer> parentValIndex)
	{
		if (parentValIndex.isEmpty())
		{
			return (counts[valIndex] + 1) 
					/ (Utils.sum(counts) + att.numValues());
		}
		double sum = 0;
		for (int i = 0; i < att.numValues(); i++)
		{
			sum += counts[getPosition(i, parentValIndex)];
		}
		return (counts[getPosition(valIndex, parentValIndex)] + 1) / 
				(sum + (double) att.numValues());
	}
	
	public double getCondProb(Instance instance, int classValIndex)
	{
		int valIndex = (int) instance.value(attIndex);
		List<Integer> parentValIndex = new ArrayList<Integer>();
		if (attIndex == instance.classIndex())
		{
			return getCondProb(classValIndex, parentValIndex);
		}
		for (int i = 0; i < parentIndexes.size() - 1; i++)
		{
			parentValIndex.add((int) instance.value(parentIndexes.get(i)) );
		}
		parentValIndex.add(classValIndex);
		return getCondProb(valIndex, parentValIndex);
	}
	
	private int getPosition(int valIndex, List<Integer> parentValIndex)
	{
		if (parentValIndex.isEmpty())
		{
			return valIndex;
		}
		int sum = 0;
		sum += digits[0] * valIndex;
		for(int i = 1; i < digits.length; i++)
		{
			sum += digits[i] * parentValIndex.get(i - 1);
		}
		return sum;
	}
}
