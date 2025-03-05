import threading
import os
import time
import glob
import sys
import queue
import random
import logging
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

from coordinator import Coordinator
from compute import Compute
from ml import ML

from compute.ttypes import Model

from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import socket

log_dir = 'logs'
log_file_path = os.path.join(log_dir, 'coordinator.log')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    filename=log_file_path,
    filemode='w'
)

def read_compute_machines(filename):
    machines = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                machine, port = line.split(',')
                machines.append((machine.strip(), int(port.strip())))

    return machines

class CoordinatorHandler:
    def __init__(self, scheduling_policy):
        self.fqueue = queue.Queue()
        self.fqueue_mutex = threading.Lock()
        self.shgradient = ([], [])
        self.init_matrices = [], []
        self.shgradient_mutex = threading.Lock()
        self._scheduling_policy = scheduling_policy

    def compute_work(self, machine, transport, compute_node, shared_weights):

        hostname = machine
        
        logging.info(f"[{hostname}] Thread started")
        print(f"[{hostname}] Thread started")
        while True:
            self.fqueue_mutex.acquire()
            try:
                fname = self.fqueue.get(block=True, timeout=3)
                self.fqueue_mutex.release()
            except queue.Empty as e:
                self.fqueue_mutex.release()
                transport.close()
                logging.debug(f"[{hostname}] Queue empty")
                print(f"[{hostname}] Queue empty")
                break
            
            if (self._scheduling_policy == 2):
                load_prob = compute_node.get_load_probability()
                rand_val = random.random()
                if rand_val < load_prob:
                    logging.warning(f'[{hostname}] Rejected task.')
                    print(f'[{hostname}] Rejected task.')
                    self.fqueue_mutex.acquire()
                    self.fqueue.put(fname)
                    self.fqueue_mutex.release()
                    continue

            logging.debug(f'[{hostname}] Training: {fname}')
            print(f'[{hostname}] Training: {fname}')
            model = Model()
            model.V = shared_weights[0]
            model.W = shared_weights[1]

            compute_node.set_model(model)           
            compute_node.train(fname)


            self.shgradient_mutex.acquire()
            new_model = compute_node.get_gradient()

            self.shgradient = (ML.sum_matricies(self.shgradient[0], new_model.V), ML.sum_matricies(self.shgradient[1], new_model.W))
            self.shgradient_mutex.release()
        logging.info(f"[{hostname}] Exiting...")
        print(f"[{hostname}] Exiting...")


    def train(self, files_dir, rounds, epochs, h, k, eta):
        unfiltered_files = os.listdir(files_dir)
        validate_files = []
        directory_path = os.getcwd() + "/ml/letters"
        
        files = os.listdir(directory_path)
        val = ""
        for i in range(len(files)):
            if files[i].startswith("validate_"):
                validate_files.append(files[i])
            

        almighty = ML.mlp()
        almighty.init_training_random(files_dir + '/' +files[0], k, h)
        self.init_matrices = almighty.get_weights()
        self.shgradient = almighty.get_weights()
        jobs = 0.0

        for i in range(rounds):

            shweights = almighty.get_weights()
            for i in range(1, len(files)):
                filename = files[i]   
                self.fqueue.put(files_dir + "/" + filename)
                jobs += 1.0
                
            if (jobs == 0.0):
                return almighty.validate(validate_files[0])
            threads = []
            machines = read_compute_machines("compute_nodes.txt")
            
            for machine, port in machines:    
                try:
                    port = int(port)
                    
                    transport = TSocket.TSocket(machine, port)
                    transport = TTransport.TBufferedTransport(transport)

                    protocol = TBinaryProtocol.TBinaryProtocol(transport)
                    compute_node = Compute.Client(protocol)

                    transport.open()
                    # portnum+=1
                    logging.info(f"Successfully connected to {machine}:{port}")
                    print(f"Successfully connected to {machine}:{port}")
                    thr = threading.Thread(target=self.compute_work, args=(machine, transport, compute_node, shweights))
                    threads.append(thr)
                    thr.start()
    


                except TTransport.TTransportException as e:

                    logging.error(f'Error connecting to {machine}::{port} with error: {str(e)}' )
                    print(f'Error connecting to {machine}::{port} with error: {str(e)}' )
                    
                except Exception as e:
                    logging.critical(f'Unexpected error: {str(e)}')
                    print(f'Unexpected error: {str(e)}')
                    break
                    
            
            for thr in threads:
                thr.join()

            if jobs == 0:
                jobs = 1
            ML.scale_matricies(self.shgradient[0], (1.0 / jobs))
            ML.scale_matricies(self.shgradient[1], (1.0 / jobs))
            almighty.update_weights(self.shgradient[0], self.shgradient[1])
            validation_err = almighty.validate(files_dir + "/" + validate_files[0])
            logging.info(f"Validation error: {validation_err}\n")
            print(f"Validation error: {validation_err}\n")
            return validation_err

        





if __name__ == '__main__':
    handler = CoordinatorHandler(int(sys.argv[2]))
    processor = Coordinator.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=int(sys.argv[1]))
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)

    # You could do one of these for a multithreaded server
    # server = TServer.TThreadedServer(
    #     processor, transport, tfactory, pfactory)
    # server = TServer.TThreadPoolServer(
    #     processor, transport, tfactory, pfactory)

    logging.info('Starting the server...')
    print('Starting the server...')
    server.serve()
    logging.info('done.')
    print('done.')
