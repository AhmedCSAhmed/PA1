from ml import ML
import sys
import glob
import socket

sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

from compute import Compute

from compute.ttypes import Model


from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer



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
        # print("THE WEIGHTS", model.V, model.W)
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
        print("Inside set_model() on server")
        
        self.model = ML.mlp()
        fname = self.get_file()
        self.matrices = shared_model.V, shared_model.W

        if not self.model.init_training_model(fname, shared_model.V, shared_model.W):
            raise ValueError("Model not initialized properly")



    
    
    
if __name__ == '__main__':
    handler = ComputeHandler()
    processor = Compute.Processor(handler)
    transport = TSocket.TServerSocket(host="0.0.0.0", port=int(sys.argv[1]))
    
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    print(f"Starting Thrift server on port {sys.argv[1]} ...")
    server.serve()