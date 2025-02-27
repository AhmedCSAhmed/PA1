from ml import ML
import sys
import glob

# sys.path.append('gen-py')
# sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

from gen_py.compute import Compute

from gen_py.compute.ttypes import Model


from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import os 


class ComputeHandler:
    def __init__(self, file="ml/letters/train_letters1.txt", eta=0.0001, epochs=75):
        self._file = file
        self._eta = eta
        self._epochs = epochs
        
    def get_file(self):
        return self._file

    def set_file(self, file):
        self._file = file

    def get_eta(self):
        return self._eta

    def set_eta(self, eta):
        self._eta = eta

    def get_epochs(self):
        return self._epochs

    def set_epochs(self, epochs):
        self._epochs = epochs
    
             
    def get_gradient(self):
        dV, dW = self.model.get_weights()
        model = Model()
        model.V = ML.calc_gradient(dV, self.matrices[0])
        model.W = ML.calc_gradient(dW, self.matrices[1])
        return model
    

    def train(self, training_file):
        """
        Parameters:
         - training_file
        """
        # No check needed defautlting to train.txt for now
        self.set_file(training_file)
        epochs = self.get_epochs()
        eta = self.get_eta()
        if self.model.train(eta, epochs) == -1:
            raise ValueError("Model Failed to execute exit -1") 
        
        print("Model successfully trained")
   
            
        

    def set_model(self, shared_model):
        """
        Parameters:
         - shared_model
         
         Under the impression our model is already intially set for us as stated in the doc
        """
        # Can handle base cases if need be as of right now we belilve it'll be the model will alawys be intialized by the main program
        self.model = ML.mlp()
        fname = self.get_file()
        self.matrices = shared_model.V, shared_model.W
        if not self.model.init_training_model(fname, shared_model.V, shared_model.W):
            raise ValueError("Model not initialized properly")
    
    
    
    def read_compute(self, file):
        lst = []
        with open(file, 'r') as f:
            for line in f:
                line.strip()
                if line:
                    name, port = line.split(',')
                    lst.append((name.strip(), int(port)))
        
        return lst
    
    
    def transportMachine(self, host, port):
        print(f"THE HOST {host} and the port {port}")
        handler = ComputeHandler()
        processor = Compute.Processor(handler)

        transport = TSocket.TServerSocket(host, port)
        tfactory = TTransport.TBufferedTransportFactory()

        pfactory = TBinaryProtocol.TBinaryProtocolFactory()


        server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

        print(f"Starting Thrift server on port {port}...")
        server.serve()
        print("GETTING INSIDE 6")

    
    
    def processEachMachine(self):
        process = []
        machines = self.read_compute("compute_nodes.txt") # Reading the mahcines
        for host, port in machines: 
            pid = os.fork() # forking for each new machine we encounter
            process.append(os.getpid()) # appending the pid id's to terminate latwr
            if pid == 0: 
                self.transportMachine(host, port) # Trying to execute this so we can connect to each machine
            else: # Failing to fork always hit's this case need to figure out why
                print("Can't execute child") 
                return -1     
        
        print(process)
        return 0     

        
   
            
            
        
        
            
            
        
        
        
                    
                    
if __name__ == "__main__":  
    handler = ComputeHandler()
    processor = Compute.Processor(handler)

    transport = TSocket.TServerSocket(host='127.0.0.1', port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    handler.processEachMachine()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    # print("Starting Thrift server on port 9090...")
    server.serve()