from ml import ML
import sys
import glob
import socket
import random
import time
import logging
import logging.handlers
import os
import multiprocessing

sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

from compute import Compute

from compute.ttypes import Model


from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

hostname = socket.gethostname()

### Logging implementation

log_dir = 'logs'
log_file = hostname + '_compute.log'
log_file_path = os.path.join(log_dir, log_file)

# Create the log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def create_log_handlers():
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    return file_handler

def setup_logging():
    log_queue = multiprocessing.Queue()  # Create a Queue for inter-process communication

    # Create a QueueListener to process logs
    listener = logging.handlers.QueueListener(log_queue, create_log_handlers())
    listener.start()

    # Configure the logger to use QueueHandler for worker processes
    handler = logging.handlers.QueueHandler(log_queue)
    logging.basicConfig(level=logging.DEBUG, handlers=[handler])
    
    return listener

def worker(log_queue, worker_id):
    logger = logging.getLogger(f"worker-{worker_id}")
    logger.info(f"Worker {worker_id} started.")
    time.sleep(1)  # Simulating some work
    logger.info(f"Worker {worker_id} finished.")




class ComputeHandler:
    def __init__(self, load_prob, file="ml/letters/train_letters1.txt", eta=0.0001, epochs=75):
        self._file = file
        self._eta = eta
        self._epochs = epochs
        self._load_probability = load_prob
        
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
        # Injecting load
        rand_val = random.random()
        if (rand_val < self._load_probability):
            logging.info(f"[{hostname}] Simulating load injection...")
            print(f"[{hostname}] Simulating load injection...")
            time.sleep(3)


        # No check needed defautlting to train.txt for now
        self.set_file(training_file)
        epochs = self.get_epochs()
        eta = self.get_eta()
        if self.model.train(eta, epochs) == -1:
            raise ValueError("Model Failed to execute exit -1") 
        
        logging.info(f"[{hostname}] Model successfully trained")
        print(f"[{hostname}] Model successfully trained")
        return True
   

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
        logging.debug("Initializing model...")
        print("Initializing model...")
        if not self.model.init_training_model(fname, shared_model.V, shared_model.W):
            logging.error(f"[{hostname}] Model not initialized properly")
            raise ValueError("Model not initialized properly")
        logging.debug(f"[{hostname}] Model has been set")
        print(f"[{hostname}] Model has been set")

    def get_load_probability(self):
        return self._load_probability



    
    
    
if __name__ == '__main__':
    listener = setup_logging()
    handler = ComputeHandler(float(sys.argv[2]))
    processor = Compute.Processor(handler)
    transport = TSocket.TServerSocket(host="0.0.0.0", port=int(sys.argv[1]))
    
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    logging.info(f"[{hostname}] Starting Thrift server on port {sys.argv[1]} ...")
    server.serve()
    listener.stop()