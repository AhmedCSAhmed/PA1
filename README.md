# Distributed ML System
The purpose of this project is to train and validate data for a distributed machine learning model. We accomplish this by offloading data to nearby compute nodes in order to train and validate data for our ML model in the most efficient way possible. 

## How to Run

1. Go to the PA1 directory and split two terminals (we will call them 1a and 1b)
  - Leave the left terminal for now, we will come back to this
  - In the right terminal, run `python3 coordinator_node.py 8080 <load_flag>` where the load_flag is 1 for random scheduling and 2 for load balancing
2. Split another 5 terminals (we will call these 2a-2e)
  - Run `python3 compute_node.py 9091 <load_prob>` in 2a where the load_flag is the load probability of the compute node
  - Open compute_nodes.txt. Notice the 5 hostnames and port numbers written
  - For the 4 non-localhost nodes, ssh into them in the following steps in terminals 2b-2e
    - Run `ssh <x500>@csel-kh1260-<machine_num>.cselabs.umn.edu.cselabs.umn.edu`
    - Complete the authentication steps
  - Go to the project directory in terminals 2b-2e
  - Run `python3 compute_node.py <portnum> <load_flag>` in 2b-2e where the load_flag is the load probability of that compute node and the portnum is the port number specified for that machine in the txt file
3. Go back to terminal 1a and run `python3 client.py` or `time python3 client.py` to time how long it takes to train the ML model.
4. Notice the server updates in the terminals, or in the corresponding log files in the `logs` directory

