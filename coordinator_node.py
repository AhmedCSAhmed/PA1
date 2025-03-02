#!/usr/bin/env python

#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
#

import threading
import os
import time
import glob
import sys
import queue
# sys.path.append('gen-py')
# sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

from gen_py.coordinator import Coordinator
from gen_py.compute import Compute
from ml import ML

from gen_py.compute.ttypes import Model

# from shared.ttypes import SharedStruct

from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import socket

    
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
    def __init__(self):
        # TODO: class variables here
        self.fqueue = queue.Queue()
        self.fqueue_mutex = threading.Lock()
        self.shgradient = ([], [])
        self.init_matrices = [], []
        self.shgradient_mutex = threading.Lock()

    def compute_work(self, transport, compute_node, shared_weights):
        
        print(f"Thread started for {compute_node}")
        time.sleep(5)  # Keep thread alive
        # print("THREAD INSTANT DEATH")
        self.fqueue_mutex.acquire() # locking this critical section
        while True:
            try:
                fname = self.fqueue.get(block=True, timeout=5)
                print("SELF MUTEX LOCK TYPE WHEN TRYING TO REALESE", self.fqueue_mutex)
            except queue.Empty as e:
                self.fqueue_mutex.release()
                transport.close()
                return -1
            
            print(f'Thread {threading.get_ident()} working on file: {fname}')
            model = Model()
            model.V = shared_weights[0]
            model.W = shared_weights[1]


            # print("THE SHARED WEIGHTS INSIDE OF COMPUTE_WORK FUNC ------>", shared_weights)
            compute_node.send_set_model(model)
            compute_node.recv_set_model()
            print("SENDING TRAIN")            
            compute_node.send_train(fname)
            compute_node.recv_train()
            print("REVIVED TRAIN")


            self.shgradient_mutex.acquire()
            # print("COMPUTE NODE.GET GRAIDENT ----->", compute_node.get_gradient())
            new_model = compute_node.get_gradient()

            self.shgradient = (ML.sum_matricies(self.shgradient[0], new_model.V), ML.sum_matricies(self.shgradient[1], new_model.W))
            self.shgradient_mutex.release()
            print(f"Exit thread: {threading.get_ident()}")


    def train(self, files_dir, rounds, epochs, h, k, eta):
        unfiltered_files = os.listdir(files_dir)
        validate_files = []
        directory_path = "/home/ahme0478/Desktop/CSCI5105/PA1/ml/letters"
        
        files = os.listdir(directory_path)
        print(len(files))
        val = ""
        for i in range(len(files)):
            if files[i].startswith("validate_"):
                validate_files.append(files[i])
            
        
        for i in validate_files:
            print("THE REMOVED FILE ---->", i)
            files.remove(i)
            
        print("THE VALIDATE FILES -->", validate_files)
        
        
      
        
                
        
        # print("Files begining of func -->", files)
        almighty = ML.mlp()
        # print("validate Files _________>>>>>>>", validate_files)
        # print("FILE DIRECTORY --->", files_dir)
        almighty.init_training_random(files_dir + '/' +files[0], k, h)
        # print("ALMIGHTY INTALIZED IS ", almighty.is_initialized())
        
        # print("ALMIGHTY WEIGHTS", almighty.get_weights())
        self.init_matrices = almighty.get_weights()
        self.shgradient = almighty.get_weights()

        jobs = 0.0
        # print("HITTTTTTT IT RIGHT NOW3")

        for i in range(rounds):
            # print("HITTTTTTT IT RIGHT NOW2")

            shweights = almighty.get_weights()
            # print("SHARED WEIGHTSSSS --->", shweights)
            
            # print("HITTTTTTT IT RIGHT NOW4")
            # print("FILESSS BEFRORE THE FOR LOOP --->", files)
            for i in range(1, len(files)):
                # print("HITTTTTTT IT RIGHT NOW5")

                filename = files[i]
                print("WORKING on", filename)
              
                
                self.fqueue.put(files_dir + "/" + filename)
                jobs += 1.0
                
            if (jobs == 0.0):
                return almighty.validate(validate_files[0])
            # print("HITTTTTTT IT RIGHT NOW2")
            threads = []
            machines = read_compute_machines("compute_nodes.txt")
            
            for machine, port in machines:    
                try:
                    
                    print("ATTEMPTING TO CONNECT THRIFT!!")
                    port = int(port)
                    
                    transport = TSocket.TSocket(machine, port)
                    print(f"CREATED SOCKET ON {machine}:{port}!")

                    transport = TTransport.TBufferedTransport(transport)
                    print(f"Wrapped transport for {machine}:{port}!")

                    protocol = TBinaryProtocol.TBinaryProtocol(transport)
                    compute_node = Compute.Client(protocol)
            
                    print(f"ABOUT TO OPEN TRANSPORT FOR {machine}:{port}!\n")

                    transport.open()
                    # portnum+=1
                    print(f"Successfully connected to port {port}")
                    print("SUCUCESSFULLY CONNECTED!!!!")
                    thr = threading.Thread(target=self.compute_work, args=(transport, compute_node, shweights))
                    threads.append(thr)
                    print("STARTING THE THREAD!!! -->")
                    thr.start()
    


                except TTransport.TTransportException as e:

                    print(f'Stopped at machine {machine}, port {port}' )
                    break
                    
                except Exception as e:
                    print(f'Unexpected error: {e}')
                    break
                    
            
            print("THE AMOUNT OF THREADS RUNNING -->", len(threads))
            for thr in threads:
                thr.join()

            if jobs == 0:
                jobs = 1
            print("GET TO VALIDATION SCALE PORTION")
            ML.scale_matricies(self.shgradient[0], (1.0 / jobs))
            ML.scale_matricies(self.shgradient[1], (1.0 / jobs))
            almighty.update_weights(self.shgradient[0], self.shgradient[1])
            validation_err = almighty.validate(files_dir + "/" + validate_files[0])
            print(f"VALIDATION ERROR --> {validation_err}\n")
            return validation_err

        





if __name__ == '__main__':
    handler = CoordinatorHandler()
    processor = Coordinator.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)

    # You could do one of these for a multithreaded server
    # server = TServer.TThreadedServer(
    #     processor, transport, tfactory, pfactory)
    # server = TServer.TThreadPoolServer(
    #     processor, transport, tfactory, pfactory)

    print('Starting the server...')
    server.serve()
    print('done.')
