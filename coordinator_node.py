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
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

from coordinator import Coordinator
from compute import Compute
from ml import ML

from compute.ttypes import Model

# from shared.ttypes import SharedStruct

from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer


class CoordinatorHandler:
    def __init__(self):
        # TODO: class variables here
        self.fqueue = queue.Queue()
        self.fqueue_mutex = threading.Lock()
        self.shgradient = []
        self.shgradient_mutex = threading.Lock()

    def compute_work(self, transport, compute_node, shared_weights):
        
        self.fqueue_mutex.acquire()
        while True:
            try:
                fname = self.fqueue.get(block=True, timeout=5)

            except queue.Empty as e:
                self.fqueue_mutex.release()
                transport.close()
                return -1
            
            self.fqueue_mutex.release()
            model = Model()
            model.V = shared_weights[0]
            model.W = shared_weights[1]

            compute_node.set_model(model)
            compute_node.train(fname)

            self.shgradient_mutex.acquire()
            shgradient = ML.sum_matrices(shgradient, compute_node.get_gradient())
            self.shgradient_mutex.release()


    def train(self, dir, rounds, epochs, h, k, eta):
        almighty = ML.mlp()
        almighty.init_training_random(dir, k, h)

        jobs = 0.0


        for i in rounds:
            shgradient = []
            shweights = almighty.get_weights()

            for filename in os.listdir(dir):
                if filename.endswith('_train.txt'):
                    file_path = os.path.join(dir, filename)
                    self.fqueue.put(file_path)
                    jobs += 1.0
            
            if (jobs == 0.0):
                return almighty.validate()

            portnum = 9090
            threads = []
            while(True):
                transport = TSocket.TSocket('localhost', portnum)
                transport = TTransport.TBufferedTransport(transport)
                protocol = TBinaryProtocol.TBinaryProtocol(transport)
                compute_node = Compute.Client(protocol)
                try:
                    transport.open()
                    portnum += 1
                    thr = threading.Thread(target=compute_work, args=(transport, compute_node, shweights))
                    threads.append(thr)
                    thr.start()

                except TTransport.TTransportException as e:
                    break
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    break

            for thr in threads:
                thr.join()

            ML.scale_matricies(shgradient, (1.0 / jobs))
            almighty.update_weights(shgradient)

            validation_err = almighty.validate()
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
