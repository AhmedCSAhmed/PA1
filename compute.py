from PA1.ml.ML import init_training_model,train

from gen_py.compute import Processor
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer



class computeService:
    def __init__(self, file="PA1/ml/letters/train_letters1.txt", eta=0.0001, epochs=75):
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
        pass
    

    def train(self, training_file):
        """
        Parameters:
         - training_file
        """
        # No check needed defautlting to train.txt for now
        self.set_file(training_file)
        epochs = self.get_epochs()
        eta = self.get_eta()
        if train(eta, epochs) == -1:
            raise ValueError("Model Failed to execute exit -1") 
        
        print("Model successfully trained")
   
            
        

    def set_model(self, shared_model):
        """
        Parameters:
         - shared_model
         
         Under the impression our model is already intially set for us as stated in the doc
        """
        # Can handle base cases if need be as of right now we belilve it'll be the model will alawys be intialized by the main program
        
        fname = self.get_file()
        init_training_model(fname, shared_model.V, shared_model.W)



    
    
    
if __name__ == '__main__':
    handler = computeService()
    processor = Processor(handler)

    transport = TSocket.TServerSocket(host='127.0.0.1', port=9091)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    print("Starting Thrift server on port 9090...")
    server.serve()