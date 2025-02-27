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
        self.fqueue_mutex.acquire() # locking this critical section
        while True:
            try:
                fname = self.fqueue.get(block=True, timeout=5)
                self.fqueue_mutex.release()
            except queue.Empty as e:
                self.fqueue_mutex.release()
                transport.close()
                return -1
            
            print(f'Thread {threading.get_ident()} working on file: {fname}')
            model = Model()
            model.V = shared_weights[0]
            model.W = shared_weights[1]

            compute_node.set_model(model)
            compute_node.train(fname)

            self.shgradient_mutex.acquire()
            new_model = compute_node.get_gradient()

            self.shgradient = (ML.sum_matricies(self.shgradient[0], new_model.V), ML.sum_matricies(self.shgradient[1], new_model.W))
            self.shgradient_mutex.release()
        print(f"Exit thread: {threading.get_ident()}")


    def train(self, files_dir, rounds, epochs, h, k, eta):
        unfiltered_files = os.listdir(files_dir)
        validate_files = [f for f in unfiltered_files if f.startswith("validate_")]
        files = [f for f in unfiltered_files if f.startswith("train_")]
        almighty = ML.mlp()
        almighty.init_training_random(files_dir + "/" + files[0], k, h)
        self.init_matrices = almighty.get_weights()
        self.shgradient = almighty.get_weights()

        jobs = 0.0


        for i in range(rounds):
            shweights = almighty.get_weights()

            for i in range(1, len(files)):
                filename = files[i]
                if filename.startswith('validate_') and filename.endswith('.txt'):
                    validate_file = filename # storing our validate file for later use
                    continue
                self.fqueue.put(files_dir + "/" + filename)
                jobs += 1.0
                
            if (jobs == 0.0):
                return almighty.validate(validate_file)

            portnum = 9091
            threads = []
            machines = read_compute_machines("compute_nodes.txt")
            
            for machine, port in machines:
                transport = TSocket.TSocket(machine, port)
                transport = TTransport.TBufferedTransport(transport)
                protocol = TBinaryProtocol.TBinaryProtocol(transport)
                compute_node = Compute.Client(protocol)
                try:
                    transport.open()
                    print(f"Successfully connected to port {port}")
                    portnum += 1
                    thr = threading.Thread(target=self.compute_work, args=(transport, compute_node, shweights))
                    threads.append(thr)
                    thr.start()

                except TTransport.TTransportException as e:
                    print(f'Stopped at machine {machine}, port {port}', port)
                    break
                except Exception as e:
                    print(f'Unexpected error: {e}')
                    break

            for thr in threads:
                thr.join()

            ML.scale_matricies(self.shgradient[0], (1.0 / jobs))
            ML.scale_matricies(self.shgradient[1], (1.0 / jobs))
            almighty.update_weights(self.shgradient[0], self.shgradient[1])

            validation_err = almighty.validate(files_dir + "/" + validate_files[0])
            print(validation_err)
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
