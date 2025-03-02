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
# "AS IS" dsfdsfBASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
#

import sys
import glob
# sys.path.append('gen-py')
# sys.path.insert(0, glob.glob('../../thrift-0.19.0/lib/py/build/lib*')[0])

from gen_py.coordinator import Coordinator

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


def main():
    # Make socket
    transport = TSocket.TSocket('localhost', 9090)

    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)

    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # Create a client to use the protocol encoder
    client = Coordinator.Client(protocol)

    # Connect!
    transport.open()


    
# # dir: Any,
#     rounds: Any,
#     epochs: Any,
#     h: Any,
#     k: Any,
#     eta: Any

    try:
        sum_ = client.train("/home/ahme0478/Desktop/CSCI5105/PA1/ml/letters" , 25, 15, 20, 26, 0.0001)
        print("train(\"/home/ahme0478/Desktop/CSCI5105/PA1/ml\" , 2, 30, 20, 26, 0.0001)\n")
        print("THE RESULT OF THE SUM FORM THE CLIENT", sum_)
        # 0.9611428571428572
    except Exception as e:
        print('Error occured: ', e)



    # Close!
    transport.close()


if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)
