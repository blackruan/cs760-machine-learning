import java.util.ArrayList;
import java.util.List;

import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;

/**
 * Assumptions:
 * (i) the class attribute is binary, 
 * (ii) it is named 'class'
 * (iii) it is the last attribute listed in the header section.
 * 
 * This class can only handle numeric and nominal attributes, and simple ARFF files.
 */
public class DecisionTree 
{
	private DecTreeNode root;
	private ArrayList<Attribute> attributes; // ordered list of all attributes

	/**
	 * Build a decision tree given a training set and "m" defined in the homework
	 * description.
	 * 
	 * @param train: the training set
	 * @param m: as defined in the homework description
	 */
	DecisionTree(Instances train, int m) 
	{
		int numOfAttributes = train.numAttributes();
		attributes = new ArrayList<Attribute>();
		List<Attribute> candidateAttributes = new ArrayList<Attribute>();
		for(int i = 0; i < numOfAttributes - 1; i++)
		{
			attributes.add(train.attribute(i));
			candidateAttributes.add(train.attribute(i));
		}
		// add class attribute
		attributes.add(train.attribute(numOfAttributes - 1));
		this.root = buildtree(train, candidateAttributes, null, 
				majorityVote(train), m);
	}
	
	/**
	 * Helper method to build a tree and return a root node.
	 * @param instances: training examples
	 * @param candidateAttributes: attributes can be used
	 * @param parentAttributeValue: answer to the parent's attribute
	 * @param majorityOfParent: majority vote label of parent node
	 * @param m: as defined in the homework description
	 */
	private DecTreeNode buildtree(Instances instances, List<Attribute> candidateAttributes, 
			String parentAttributeValue, String majorityOfParent, int m)
	{
		// counts of each label of class attribute among instances
		int[] occurrences = new int[instances.numClasses()];
		
		// stopping criteria: the number of examples that reached the leaf node is 0, 
		// we output the majority value of the parent examples
		if (instances.isEmpty())
		{
			for (int i = 0; i < occurrences.length; i++)
			{
				occurrences[i] = 0;
			}
			return new DecTreeNode(majorityOfParent, null, parentAttributeValue,
					true, occurrences);
		}
		
		int indexOfMajorityVote = 0;
		AttributeStats classStats =  instances.attributeStats(instances.classIndex());
		occurrences = classStats.nominalCounts;
		for (int i = 0; i < occurrences.length; i++)
		{
			// stopping criteria: (i) all of the training instances reaching the 
			// node belong to the same class
			if (occurrences[i] == instances.size())
			{
				return new DecTreeNode(instances.classAttribute().value(i), 
						null, parentAttributeValue, true, occurrences);
			}
			if (occurrences[i] > occurrences[indexOfMajorityVote])
			{
				// If the classes of the training instances reaching a leaf are 
				// equally represented, the leaf would predict the first class 
				// listed in the ARFF file.
				indexOfMajorityVote = i;
			}
		}
		
		// stopping criteria: (ii) there are fewer than m training instances 
		// reaching the node	
		if (instances.size() < m)
		{
			return new DecTreeNode(instances.classAttribute().value(indexOfMajorityVote), 
					null, parentAttributeValue, true, occurrences);
		}
		
		// stopping criteria: (iv)there are no more remaining candidate splits at the node
		if (candidateAttributes.isEmpty())
		{
			return new DecTreeNode(instances.classAttribute().value(indexOfMajorityVote), 
					null, parentAttributeValue, true, occurrences);
		}
		
		// Candidate splits
		List<Split> candidateSplits = determineCandidateSplits(instances, candidateAttributes);
		
		// stopping criteria: (iv)there are no more remaining candidate splits at the node
		if (candidateSplits.isEmpty())
		{
			return new DecTreeNode(instances.classAttribute().value(indexOfMajorityVote), 
					null, parentAttributeValue, true, occurrences);
		}
				
		Split bestSplit = findBestSplit(instances, candidateSplits);
		
		// stopping criteria: (iii) no feature has positive information gain
		if (bestSplit == null)
		{
			 return new DecTreeNode(instances.classAttribute().value(indexOfMajorityVote), 
						null, parentAttributeValue, true, occurrences);
		}
		
		// Create internal node. Store majorityVote in the label parameter.
		DecTreeNode root = new DecTreeNode(instances.classAttribute().value(indexOfMajorityVote),
				bestSplit.attribute, parentAttributeValue, false, occurrences);
		if (bestSplit.attribute.isNumeric())
		{
			for (int i = 0; i < 2; i++)
			{
				// left branch, corresponding to "<="
				if (i == 0)
				{
					root.addChild(buildtree(subInstances(bestSplit, i, instances),
							candidateAttributes, Double.toString(bestSplit.threshold), 
							instances.classAttribute().value(indexOfMajorityVote), m));
				}
				// right branch, corresponding to ">"
				if (i == 1)
				{
					root.addChild(buildtree(subInstances(bestSplit, i, instances),
							candidateAttributes, Double.toString(bestSplit.threshold), 
							instances.classAttribute().value(indexOfMajorityVote), m));
				}
			}
		}
		// bestSplit attribute is nominal
		else
		{
			List<Attribute> subCandidateAttributes = new ArrayList<Attribute>();
			// form a subAttributes list without bestAttribute
			for (Attribute attribute : candidateAttributes)
			{
				if (!attribute.equals(bestSplit.attribute))
				{
					subCandidateAttributes.add(attribute);
				}
			}			
			for (int i = 0; i < bestSplit.attribute.numValues(); i++)
			{
				root.addChild(buildtree(subInstances(bestSplit, i, instances),
						subCandidateAttributes, bestSplit.attribute.value(i), 
						instances.classAttribute().value(indexOfMajorityVote), m));
			}
		}
		return root;
	}

	/**
	 * Determine candidate splits for the input instances
	 * @param instances
	 * @param candidateAttributes: candidate attributes that can be used to split
	 */
	private List<Split> determineCandidateSplits(Instances instances, List<Attribute> candidateAttributes)
	{
		List<Split> candidateSplits = new ArrayList<Split>();
		for(Attribute candidate : candidateAttributes)
		{
			if(candidate.isNominal())
			{
				candidateSplits.add(new Split(candidate));
			}
			else
			{
				List<Double> thresholds = determineCandidateNumericSplits(instances, 
						candidate);
				for (Double threshold : thresholds)
				{
					candidateSplits.add(new Split(candidate, threshold));
				}
			}
		}
		return candidateSplits;
	}
	
	/**
	 * Determine candidate list of threshold to split for numeric features
	 * @param instances
	 * @param attribute: numeric feature needed to get threshold
	 */
	private List<Double> determineCandidateNumericSplits(Instances instances, 
			Attribute attribute)
	{
		// follow algorithm in the slides
		List<Double> thresholds = new ArrayList<Double>();
		List<List<Instance>> sets = new ArrayList<List<Instance>>();
		instances.sort(attribute);
		List<Instance> initialSet = new ArrayList<Instance>();
		initialSet.add(instances.get(0));
		Double value = instances.get(0).value(attribute);
		sets.add(initialSet);
		// divide instances into groups according to their value
		for(int i = 1; i < instances.numInstances(); i++)
		{
			Instance instance = instances.get(i);
			if(instance.value(attribute) == value)
			{
				sets.get(sets.size() - 1).add(instance);
			}
			else
			{
				List<Instance> set = new ArrayList<Instance>();
				set.add(instance);
				sets.add(set);
				value = instance.value(attribute);
			}
		}
		// determine whether to add midpoints between each pair of group
		for(int i = 0; i < sets.size() - 1; i++)
		{
			boolean flag = false;
			String label = sets.get(i).get(0).stringValue(instances.classIndex());
			
			for(int j = 1; j < sets.get(i).size(); j++)
			{
				if (!sets.get(i).get(j).stringValue(instances.classIndex()).equals(label))
				{
					thresholds.add((sets.get(i).get(0).value(attribute) + 
							sets.get(i + 1).get(0).value(attribute)) / 2 );
					flag = true;
					break;
				}
			}
			if (flag)
			{
				continue;
			}
			for(int j = 0; j < sets.get(i + 1).size(); j++)
			{
				if (!sets.get(i + 1).get(j).stringValue(instances.classIndex()).equals(label))
				{
					thresholds.add((sets.get(i).get(0).value(attribute) + 
							sets.get(i + 1).get(0).value(attribute)) / 2 );
					break;
				}
			}
		}
		return thresholds;
	}
	
	/**
	 * Find the best split of input instances
	 * @param instances
	 * @param candidateSplits: candidate list of splits to find the best split
	 */
	private Split findBestSplit(Instances instances, List<Split> candidateSplits) 
	{
		double bestGain = -10;
		Split bestSplit = null;
		double rootEntropy;
		AttributeStats labelStats = instances.attributeStats(instances.classIndex());
		int[] labelCounts = labelStats.nominalCounts;
		rootEntropy = calculateEntropy(labelCounts);
		for(Split split : candidateSplits)
		{
			double gain = calculateGain(split, instances, rootEntropy);
			// When there is a tie between two features in their information gain, 
			// break the tie in favor of the feature listed first in the header 
			// section of the ARFF file.
			
			// When there is a tie between two different thresholds for a numeric 
			// feature, break the tie in favor of the smaller threshold.
			if (gain > bestGain)
			{
				bestGain = gain;
				bestSplit = split;
			}	
		}
		// no feature has positive information gain
		if (bestGain <= 0)
		{
			return null;
		}
		return bestSplit;
	}
	
	/**
	 * Return sub instances according to the split
	 * @param split: split for the input instances
	 * @param i: index of answer for the split
	 * @param instances
	 */
	private Instances subInstances (Split split, int i, Instances instances)
	{
		Instances sub = new Instances(instances.relationName(), this.attributes, 
				instances.size());
		sub.setClassIndex(instances.classIndex());
		if (split.attribute.isNumeric())
		{
			// corresponding to "<="
			if(i == 0)
			{
				for (Instance instance : instances)
				{
					if (instance.value(split.attribute) <= split.threshold)
					{
						sub.add(instance);
					}
				}
			}
			// corresponding to ">"
			else
			{
				for (Instance instance : instances)
				{
					if (instance.value(split.attribute) > split.threshold)
					{
						sub.add(instance);
					}
				}
			}
		}
		// split attribute is nominal
		else
		{
			for (Instance instance : instances)
			{
				if (instance.stringValue(split.attribute).equals(split.attribute.value(i)))
				{
					sub.add(instance);
				}
			}
		}
		return sub;
	}
	
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() 
	{

		printTreeNode(root, null, 0, 0);
	}
	
	/**
	 * Prints the subtree of the node
	 * with each line prefixed by "|" + tab * k spaces.
	 */
	private void printTreeNode(DecTreeNode p, DecTreeNode parent, int k, int indexOfBranch) 
	{
		int index = 0;
		// the node is root 
		if (parent == null) 
		{
			for (DecTreeNode child : p.children)
			{
				printTreeNode(child, p, 0, index);
				index ++;
			}
			return;
		} 
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) 
		{
			sb.append("|\t");
		}
		Attribute parentAttribute = parent.attribute;
		String parentAttributeName = parentAttribute.name();
		sb.append(parentAttributeName);
		
		if (parentAttribute.isNumeric()) 
		{
			if (indexOfBranch == 0)
			{
				sb.append(" <= ");
			}
			else
			{
				sb.append(" > ");
			}
			System.out.print(sb.toString());
			double parentAttributeValue = Double.parseDouble(p.parentAttributeValue);
			System.out.format("%f", parentAttributeValue);
		}
		// the attribute is nominal
		else
		{
			sb.append(" = ");
			String parentAttributeValue = p.parentAttributeValue;
			sb.append(parentAttributeValue);
			System.out.print(sb.toString());
		}
		
		StringBuilder s = new StringBuilder();
		s.append(" [" + p.numOfInstances[0] + " " + p.numOfInstances[1] + "]");
		
		if (p.terminal) 
		{
			s.append(": " + p.label);
			System.out.println(s.toString());
		} 
		else 
		{
			System.out.println(s.toString());
			for(DecTreeNode child: p.children) 
			{
				printTreeNode(child, p, k+1, index);
				index ++;
			}
		}
	}

	/**
	 * Return the label, which most instances have
	 * 
	 */
	private String majorityVote(Instances instances)
	{
		AttributeStats classStats =  instances.attributeStats(instances.classIndex());
		int[] occurrences = classStats.nominalCounts;
		int indexOfMajorityVote = 0;
		for (int i = 0; i < occurrences.length; i++)
		{
			// always choose small index when ties
			
			// If the classes of the training instances reaching a leaf are equally
			// represented, the leaf would predict the first class listed in the ARFF file.
			if (occurrences[i] > occurrences[indexOfMajorityVote])
			{
				indexOfMajorityVote = i;
			}
		}
		return instances.classAttribute().value(indexOfMajorityVote);
	}
	
	/**
	 * Calculate gain
	 */
	private double calculateGain(Split s, Instances instances, double rootEntropy)
	{	
		double condEntropy = 0;
		if (s.attribute.isNominal())
		{
			for (int i = 0; i < s.attribute.numValues(); i++)
			{
				int[] occurrences = new int[instances.numClasses()];
				double count = 0;
				for (Instance instance : instances)
				{
					if (instance.stringValue(s.attribute).equals(s.attribute.value(i)))
					{
						occurrences[instances.classAttribute().indexOfValue(
								instance.stringValue(instances.classAttribute()) )] ++;
						count ++;
					}
				}
				condEntropy += count / instances.size() * calculateEntropy(occurrences);
			}			
		}
		// attribute is numeric
		else
		{
			for (int i = 0; i < 2; i++)
			{
				int[] occurrences = new int[instances.numClasses()];
				double count = 0;
				for (Instance instance : instances)
				{
					// corresponding to "<="
					if (i == 0)
					{
						if (instance.value(s.attribute) <= s.threshold)
						{
							occurrences[instances.classAttribute().indexOfValue(
									instance.stringValue(instances.classAttribute()) )] ++;
							count ++;
						}
					}
					// corresponding to ">"
					else
					{
						if (instance.value(s.attribute) > s.threshold)
						{
							occurrences[instances.classAttribute().indexOfValue(
									instance.stringValue(instances.classAttribute()) )] ++;
							count ++;
						}
					}
				}
				condEntropy += count / instances.size() * calculateEntropy(occurrences);
			}
			
		}
		return rootEntropy - condEntropy;
	}
	
	/**
	 * Calculate entropy
	 */
	private double calculateEntropy(int[] occurrences)
	{
		double total = 0;
		for (int i = 0; i < occurrences.length; i++)
		{
			total += (double) occurrences[i];
		}
		if (total == 0)
		{
			return 0;
		}
		double entropy = 0;
		for (int i = 0; i < occurrences.length; i++)
		{
			if (occurrences[i] > 0)
			{
				entropy += (-1) * (occurrences[i] / total) * 
						(Math.log(occurrences[i] / total) / Math.log(2));
			}
		}
		return entropy;
	}
	
	public String classify(Instance instance) 
	{
		DecTreeNode node = this.root;
		while (!node.terminal)
		{
			if (node.attribute.isNominal())
			{
				String answer = instance.stringValue(node.attribute);
				int indexOfAnswer = node.attribute.indexOfValue(answer);
				node = node.children.get(indexOfAnswer);
			}
			else
			{
				double answer = instance.value(node.attribute);
				int indexOfAnswer;
				if (answer <= Double.parseDouble(node.children.get(0).parentAttributeValue))
				{
					indexOfAnswer = 0;
				}
				else
				{
					indexOfAnswer = 1;
				}
				node = node.children.get(indexOfAnswer);
			}
		}
		return node.label;
	}
}

