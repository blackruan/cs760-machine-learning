import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;

/**
 * Tree node for decision tree
 * 
 */
public class DecTreeNode 
{
	String label; 
	Attribute attribute;
	String parentAttributeValue; // if is the root, set to "-1"
	boolean terminal;
	int[] numOfInstances;
	List<DecTreeNode> children;

	DecTreeNode(String _label, Attribute _attribute, String _parentAttributeValue, 
			boolean _terminal, int[] _numOfInstances) 
	{
		label = _label;
		attribute = _attribute;
		parentAttributeValue = _parentAttributeValue;
		terminal = _terminal;
		if (_terminal) 
		{
			children = null;
		} 
		else 
		{
			children = new ArrayList<DecTreeNode>();
		}
		numOfInstances = _numOfInstances.clone();
	}

	/**
	 * Add child to the node.
	 * 
	 * For printing to be consistent, children should be added
	 * in order of the attribute values as specified in the
	 * dataset.
	 */
	public void addChild(DecTreeNode child) 
	{
		if (children != null) 
		{
			children.add(child);
		}
	}
}
