import weka.core.Attribute;

public class Split 
{
	public Attribute attribute;
	public Double threshold;
	
	Split(Attribute _attribute)
	{
		attribute = _attribute;
		threshold = null;
	}
	
	Split(Attribute _attribute, Double _threshold)
	{
		attribute = _attribute;
		threshold = _threshold;
	}
}
