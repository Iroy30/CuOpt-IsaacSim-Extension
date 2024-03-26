# cuOpt Microservice Examples
This extension enables a set of UI based examples demonstrating the cuOpt microservice for routing optimization.  These examples include:

1. Cost Matrix Example : A simple example with randomly generated routing problems represented by simple geometry. This extension directly expresses the  optimization problem to be solved as a cost matrix (matrix of pairwise costs of travel between locations). 
  
2. Waypoint Graph Example : This example leverages a waypoint graph to represent the cost of travel within the environment. In addition, this example leverages omni.cuopt.visualization to process and visualize the waypoint graph.
  
3. Intra-Warehouse Transport Demo :  This example builds upon the Waypoint Graph Example and demonstrates a more complex waypoint graph in the context of an intra-warehouse goods/equipment transport scenario.

### Requirements
To use this extension you must have access to a running instance of the cuOpt microservice (local or remote). Detailed instructions on downloading and running the cuOpt microservice can be found [here](https://github.com/NVIDIA/cuOpt-Resources#setup)


### Recommended Use
This extension demonstrates a small subset of features of the cuOpt microservice. The code for this extension is made available and should be extended for specific use cases when needed. A complete list of available features can be found in the [cuOpt docs](https://docs.nvidia.com/cuopt/)