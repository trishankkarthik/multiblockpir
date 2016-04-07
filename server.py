#!/usr/bin/env python


"""
A server which simply serves blocks to clients forever.
"""


import logging
import sys

import libserver


if __name__ == "__main__":
  line_format = "[%(process)s] [%(asctime)s] [%(name)s] [%(levelname)s] [%(funcName)s:%(lineno)s@%(filename)s] %(message)s"
  logging.basicConfig(filename="server.log", format=line_format, level=logging.DEBUG)
  address = sys.argv[1]
  port = int(sys.argv[2], 10)
  filename = sys.argv[3]

  libserver.run_forever(address, port, filename)


