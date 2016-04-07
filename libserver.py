"""
A few utilities for the convenient implementation of a LukePIR server/mirror.
"""





from __future__ import division

import SocketServer
import array
import logging
import re
import signal
import socket
import sys
import threading
import timeit

import libmultiblockpir





# GLOBAL CONSTANTS

N = None
S = libmultiblockpir.S


# GLOBAL VARIABLES

# The total number of blocks we have served.
total_number_of_blocks_served = 0

# For each block request, the turnaround time is the time spent to receive the
# request + time spent to compute the sum + time spent to send the response.

# This is the sum of the CPU wall time for every block request.
total_get_block_cpu_wall_time = 0

# This is how we safely serialize concurrent access to server statistics.
server_lock = threading.Lock()





class Commands(object):
  GET_N_COMMAND = "GET_N()"
  GET_S_COMMAND = "GET_S()"

  # GET_BLOCK((finite field order), (length of encoded array of bytes))
  # TODO: check whether encoding for array of bytes contains parenthesis!
  GET_BLOCK_COMMAND = "GET_BLOCK\((2|256), (\d+)\)"





class RequestHandler(SocketServer.StreamRequestHandler):

  CRLF = "\r\n"


  def handle_get_N_command(self, command):
    logging.debug(command)
    response = str(N)
    return response


  def handle_get_S_command(self, command):
    logging.debug(command)
    response = str(S)
    return response


  def multiply_and_add_with_database(self, field_order, V_string):
    """B (1xs) = V (1xN) * D (NxS)"""

    V_array = array.array('B')
    V_array.fromstring(V_string)
    assert len(V_array) == N
    V = (libmultiblockpir.array_of_bytes(N))(*V_array)

    B = libmultiblockpir.empty_array(S)

    assert field_order in (2, 256)
    if field_order == 2:
      libmultiblockpir.multiply_and_add_with_database_in_GF2(V, B)
    else:
      libmultiblockpir.multiply_and_add_with_database_in_GF256(V, B)

    logging.debug("BEFORE B_string")
    # http://stackoverflow.com/a/3470652
    B_string = array.array('B', B).tostring()
    logging.debug("AFTER B_string")
    return B_string


  def update_statistics(self, block_request_cpu_wall_time):
    """Thread-safe statistics."""

    global total_get_block_cpu_wall_time
    global total_number_of_blocks_served

    server_lock.acquire()

    try:
      total_get_block_cpu_wall_time += block_request_cpu_wall_time
      total_number_of_blocks_served += 1

    except:
      logging.exception("Could not update statistics!")
      raise

    finally:
      server_lock.release()


  def handle_get_block_command(self, command):
    """This function will return a non-empty response string only if the
    command is a valid GET_BLOCK command."""

    match = re.match(Commands.GET_BLOCK_COMMAND, command, flags=re.DOTALL)
    response = ''

    if match:
      logging.debug(command)

      field_order = int(match.group(1), 10)
      V_string = ''
      V_string_length = int(match.group(2), 10)

      while V_string_length > 0:
        V_string_suffix = self.rfile.readline()
        V_string += V_string_suffix
        V_string_length -= len(V_string_suffix)

      V_string = V_string[:N] # Strip out CRLF

      # Start the clock.
      start_get_block_cpu_wall_time = timeit.default_timer()

      # Actually computation of the block.
      B_string = self.multiply_and_add_with_database(field_order, V_string)

      # Stop the clock and measure the time difference.
      stop_get_block_cpu_wall_time = timeit.default_timer()
      get_block_cpu_wall_time = \
        stop_get_block_cpu_wall_time-start_get_block_cpu_wall_time

      B_string_length = len(B_string)
      response = "{0}{1}{2}".format(B_string_length,
                                    RequestHandler.CRLF,
                                    B_string)

      self.update_statistics(get_block_cpu_wall_time)

    else:
      logging.warn("Unknown command: {command}".format(command=command))

    return response


  def handle(self):
    command = self.rfile.readline().strip()
    response = ''

    try:
      if command == Commands.GET_N_COMMAND:
        response = self.handle_get_N_command(command)

      elif command == Commands.GET_S_COMMAND:
        response = self.handle_get_S_command(command)

      else:
        response = self.handle_get_block_command(command)

    except:
      logging.exception("Exception while parsing command: "+\
                        "{command}".format(command=command))

    finally:
      self.wfile.write(response+RequestHandler.CRLF)





class ParallelServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
  """Use this server to concurrently handle block requests."""

  # We set this to allow the server to instantly bind to the same address.
  # http://docs.python.org/2/library/socketserver.html#SocketServer.BaseServer.allow_reuse_address
  allow_reuse_address = True





class SerialServer(SocketServer.TCPServer):
  """Use this server to serially handle block requests."""

  # We set this to allow the server to instantly bind to the same address.
  # http://docs.python.org/2/library/socketserver.html#SocketServer.BaseServer.allow_reuse_address
  allow_reuse_address = True





def handle_interrupt_signal(signal, frame):
  if total_number_of_blocks_served > 0:
    average_get_block_cpu_wall_time = \
      total_get_block_cpu_wall_time/total_number_of_blocks_served

    logging.info("Total GET_BLOCK CPU wall time: {0}".format(
                 total_get_block_cpu_wall_time))

    logging.info("Total number of blocks served: {0}".format(
                 total_number_of_blocks_served))

    logging.info("Average GET_BLOCK CPU wall time: {0}".format(
                 average_get_block_cpu_wall_time))
  else:
    logging.info("No blocks were served.")

  sys.exit(0)





def run_forever(address, port, filename, concurrent=True):
  global N, S

  N = libmultiblockpir.memorize_database(filename)
  S = libmultiblockpir.S

  signal.signal(signal.SIGINT, handle_interrupt_signal)

  if concurrent:
    server = ParallelServer((address, port), RequestHandler)
    logging.info("Instantiated parallel server.")
  else:
    server = SerialServer((address, port), RequestHandler)
    logging.info("Instantiated serial server.")

  logging.info("Server loop running on {0}:{1}".format(address, port))
  server.serve_forever()
