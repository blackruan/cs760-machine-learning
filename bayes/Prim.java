public class Prim 
{
	public boolean[][] adjMatrix;
	
	private int numVertices;
	
	public Prim()
	{
		adjMatrix = null;
		numVertices = 0;
	}
	
	public void primAlg(double[][] _adjMatrix, int root)
	{
		numVertices = _adjMatrix.length;
		adjMatrix = new boolean[numVertices][numVertices];
		
		boolean[] settled = new boolean[numVertices];
		boolean[] unsettled = new boolean[numVertices];
		for (int i = 0; i < unsettled.length; i++)
		{
			unsettled[i] = true;
		}
		settled[root] = true;
		unsettled[root] = false;
		
		int numSettled = 1;
		while(numSettled < numVertices)
		{
			double maxWeight = -1;
			int _u = -1;
			int _v = -1;
			for(int u = 0; u < settled.length; u++)
			{
				if (settled[u] == true)
				{
					for(int v = 0; v < unsettled.length; v++)
					{
						if (unsettled[v] == true)
						{
							if(_adjMatrix[u][v] > maxWeight)
							{
								maxWeight = _adjMatrix[u][v];
								_u = u;
								_v = v;
							}
						}
					}
				}
			}
			numSettled++;
			settled[_v] = true;
			unsettled[_v] = false;
			adjMatrix[_u][_v] = true;
		}
	}
}
