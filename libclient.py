"""
A few utilities for the convenient implementation of a LukePIR client.
"""


from __future__ import division
import array
import logging
import math
import random
import socket
import threading
import timeit

import compatibility
import libmultiblockpir
import libserver
import py_ecc.ffield
import py_ecc.genericmatrix


################################ GLOBAL CLASSES ################################


class SimpleClient(object):
  """Client support class for simple Internet protocols.
  http://effbot.org/zone/socket-intro.htm"""

  CRLF = "\r\n"

  def __init__(self, host, port):
    """Connect to an Internet server."""

    try:
      self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.sock.connect((host, port))
      self.file = self.sock.makefile("rb") # buffered
    except:
      logging.exception("Failed to connect to {0}:{1}".format(host, port))
      raise

  def writeline(self, line):
    """Send a line to the server."""

    self.sock.send(line + SimpleClient.CRLF) # unbuffered write

  def read(self, maxbytes = None):
    """Read data from server."""

    if maxbytes is None:
        return self.file.read()
    else:
        return self.file.read(maxbytes)

  def readline(self):
    """Read a line from the server.  Strip trailing CR and/or LF."""

    s = self.file.readline()
    if not s:
        raise EOFError
    if s[-2:] == SimpleClient.CRLF:
        s = s[:-2]
    elif s[-1:] in SimpleClient.CRLF:
        s = s[:-1]
    return s


class Mirror(object):
  """The interface to a remote mirror."""

  def __init__(self, address, port):
    self.address = address
    self.port = port

    # Internal statistics
    # This is how we safely serialize concurrent access to statistics
    self.__mirror_lock = threading.Lock()
    self.__network_time_to_multiply_and_add_with_database = 0
    self.__number_of_blocks_multiplied_and_added = 0


  def __repr__(self):
    return "Mirror({0}, {1})".format(self.address, self.port)


  def get_client(self):
    return SimpleClient(self.address, self.port)


  def get_N(self):
    client = self.get_client()
    client.writeline(libserver.Commands.GET_N_COMMAND)
    N_string = client.readline()
    N = int(N_string, 10)
    return N


  def get_S(self):
    client = self.get_client()
    client.writeline(libserver.Commands.GET_S_COMMAND)
    S_string = client.readline()
    S = int(S_string, 10)
    return S


  def get_statistics(self):
    """Return (network_time_to_mix_blocks, number_of_mixed_blocks)."""

    # thread-safe statistics
    self.__mirror_lock.acquire()
    try:
      return (self.__network_time_to_multiply_and_add_with_database,
              self.__number_of_blocks_multiplied_and_added)
    except:
      logging.exception("{0} could not read its statistics!".format(self))
      raise
    finally:
      self.__mirror_lock.release()


  def multiply_and_add_with_database(self, W, V, S):
    """B (1xs) = V (1xN) * D (NxS), in GF(W)"""

    start_wall_time = timeit.default_timer()

    client = self.get_client()
    V_string = array.array('B', V).tostring()
    command = "GET_BLOCK({0}, {1})".format(W, len(V_string))
    client.writeline(command)
    client.writeline(V_string)

    # read by length
    B_string_length = client.file.readline()

    assert len(B_string_length) > 0
    B_string = ''
    B_string_length = int(B_string_length, 10)

    while B_string_length > 0:
      B_string_suffix = client.file.readline()
      B_string += B_string_suffix
      B_string_length -= len(B_string_suffix)

    B_string = B_string[:S]  # Strip out CRLF
    B_array = array.array('B')
    B_array.fromstring(B_string)
    B = (libmultiblockpir.array_of_bytes(S))(*B_array)

    stop_wall_time = timeit.default_timer()
    wall_time = stop_wall_time-start_wall_time

    # thread-safe statistics
    self.__mirror_lock.acquire()
    try:
      self.__network_time_to_multiply_and_add_with_database += wall_time
      self.__number_of_blocks_multiplied_and_added += 1
    except:
      logging.exception("{0} could not update its statistics!".format(self))
      raise
    finally:
      self.__mirror_lock.release()

    return B





class MirrorController(object):
  """The interface to a set of remote mirrors."""


  class InsufficientMirrorsError(RuntimeError): pass


  def __init__(self):
    # mirrors are indexed by netloc (address:port)
    # these are the Mixed Block (MB) mirrors
    self.__MB_mirrors = {}
    # and these are the Noisy Block (NB) mirrors
    self.__NB_mirrors = {}

    # parameters that must be shared by this set of mirrors
    self.N = None
    self.S = None


  def assert_mirror_consistency(self, mirror):
    """Use this to determine that:

      1. Mirrors describe exactly the same database.
      2. Mirrors use a compatible libmultiblockpir (e.g. same S)."""

    N = mirror.get_N()
    S = mirror.get_S()

    if self.N is None and self.S is None:
      self.N = N
      self.S = S

    # Every mirror must have the same N and S.
    assert self.N == N
    assert self.S == S

    logging.info("N = {0}".format(self.N))
    logging.info("S = {0}".format(self.S))


  # TODO: cache N
  def get_N(self):
    """Does every mirror agree on the same N?"""

    return self.N

  # TODO: cache S
  def get_S(self):
    """Does every mirror agree on the same S, and is it compatible with our
    S?"""

    return self.S


  def get_number_of_MB_mirrors(self):
    """Get number of MB mirrors."""

    return len(self.__MB_mirrors)


  def get_number_of_NB_mirrors(self):
    """Get number of NB mirrors."""

    return len(self.__NB_mirrors)


  def get_number_of_mirrors(self):
    """Get number of all mirrors."""

    return self.get_number_of_MB_mirrors() + self.get_number_of_NB_mirrors()


  def get_MB_mirrors(self, n=None, recycle_mirrors=False):
    """Get a random sample of n MB mirrors.

    WARNING: If recycle_mirrors is True, we will sample mirrors with replacement
    if we find k or m lacking with respect to the observed number of mirrors.
    This will defeat k-safety and lead to direct loss of privacy! This option
    is in place only for internal evaluation of the protocol and will be
    removed later."""

    if n is None:
      logging.warn("Sampling all MB mirrors")
      n = len(self.__MB_mirrors)

    if n <= self.get_number_of_MB_mirrors():
      return random.sample(self.__MB_mirrors.values(), n)
    else:
      if recycle_mirrors:
        logging.warn("Recycling MB mirrors")

        # slightly biased sample, but acceptable for our purposes
        expected_to_observed_multiplier = math.trunc(math.ceil(n/self.get_number_of_MB_mirrors()))
        MB_mirrors_population = self.__MB_mirrors.values() * expected_to_observed_multiplier
        assert n <= len(MB_mirrors_population)
        MB_mirrors = random.sample(MB_mirrors_population, n)

        logging.warn("Recycled MB mirrors = {0}".format(MB_mirrors))

        return MB_mirrors
      else:
        raise InsufficientMirrorsError()


  def get_NB_mirrors(self, n=None, recycle_mirrors=False):
    """Get a random sample of n NB mirrors.

    WARNING: If recycle_mirrors is True, we will sample mirrors with replacement
    if we find k or m lacking with respect to the observed number of mirrors.
    This will defeat k-safety and lead to direct loss of privacy! This option
    is in place only for internal evaluation of the protocol and will be
    removed later."""

    if n is None:
      logging.warn("Sampling all NB mirrors")
      n = len(self.__NB_mirrors)

    if n <= self.get_number_of_NB_mirrors():
      return random.sample(self.__NB_mirrors.values(), n)
    else:
      if recycle_mirrors:
        logging.warn("Recycling NB mirrors")

        # slightly biased sample, but acceptable for our purposes
        expected_to_observed_multiplier = math.trunc(math.ceil(n/self.get_number_of_NB_mirrors()))
        NB_mirrors_population = self.__NB_mirrors.values() * expected_to_observed_multiplier
        assert n <= len(NB_mirrors_population)
        NB_mirrors = random.sample(NB_mirrors_population, n)

        logging.warn("Recycled NB mirrors = {0}".format(NB_mirrors))

        return NB_mirrors
      else:
        raise InsufficientMirrorsError()


  def register_MB_mirror(self, address, port, recycle_mirrors=False):
    """Register an MB mirror.

    WARNING: If recycle_mirrors is True, we will sample mirrors with
    replacement if we find k or m lacking with respect to the observed number
    of mirrors. This will defeat k-safety and lead to direct loss of privacy!
    This option is in place only for internal evaluation of the protocol and
    will be removed later."""

    netloc = "{0}:{1}".format(address, port)

    assert netloc not in self.__MB_mirrors
    if recycle_mirrors is False:
      assert netloc not in self.__NB_mirrors
    else:
      logging.warn("Ignoring whether MB intersects with NB.")

    mirror = Mirror(address, port)
    self.assert_mirror_consistency(mirror)
    self.__MB_mirrors[netloc] = mirror
    logging.info("Registered {0}".format(mirror))


  def register_NB_mirror(self, address, port, recycle_mirrors=False):
    """Register an NB mirror.

    WARNING: If recycle_mirrors is True, we will sample mirrors with
    replacement if we find k or m lacking with respect to the observed number
    of mirrors. This will defeat k-safety and lead to direct loss of privacy!
    This option is in place only for internal evaluation of the protocol and
    will be removed later."""

    netloc = "{0}:{1}".format(address, port)

    if recycle_mirrors is False:
      assert netloc not in self.__MB_mirrors
    else:
      logging.warn("Ignoring whether NB intersects with MB.")
    assert netloc not in self.__NB_mirrors

    mirror = Mirror(address, port)
    self.assert_mirror_consistency(mirror)
    self.__NB_mirrors[netloc] = mirror
    logging.info("Registered {0}".format(mirror))





class SerialMBWorker(object):
  """A serial method to download a Mixed Block (MB)."""

  def __init__(self, CL, MB, S, W, i, mirror):
    self.CL = CL
    self.MB = MB
    self.S = S
    self.W = W
    self.i = i
    self.mirror = mirror


  def work(self):
    try:
      mixed_block = self.mirror.multiply_and_add_with_database(self.W, self.CL,
                                                               self.S)
      self.MB[self.i] = mixed_block
    except:
      logging.exception("SerialMBWorker({0}) aborted!".format(self.i))
      raise
    else:
      logging.info("SerialMBWorker({0}) passed".format(self.i))





class ParallelMBWorker(SerialMBWorker, threading.Thread):
  """A concurrent method to download a Mixed Block (MB)."""

  def __init__(self, CL, MB, S, W, barrier, i, mirror):
    threading.Thread.__init__(self)
    self.daemon = True

    SerialMBWorker.__init__(self, CL, MB, S, W, i, mirror)
    self.barrier = barrier


  def run(self):
    try:
      self.work()
    except:
      self.barrier.abort()
      logging.exception("ParallelMBWorker({0}) aborted the barrier!".format(self.i))
    else:
      self.barrier.wait()
      logging.info("ParallelMBWorker({0}) passed the barrier".format(self.i))





class SerialNBWorker(object):
  """A serial method to download a Noisy/New Block (NB)."""


  # This is how we safely serialize concurrent access to statistics
  __shared_lock = threading.Lock()
  # Internal statistics
  __total_encoding_and_decoding_time = 0
  __total_number_of_noisy_blocks = 0


  def __init__(self, CR, GE, N, RR, S, W, block_index, fetched_blocks, n,
               mirror, multiply_and_add_vector_with_scalar):
    self.CR = CR
    self.GE = GE
    self.N = N
    self.RR = RR
    self.S = S
    self.W = W
    self.block_index = block_index
    self.fetched_blocks = fetched_blocks
    self.n = n
    self.mirror = mirror
    self.multiply_and_add_vector_with_scalar = \
      multiply_and_add_vector_with_scalar


  # thread-safe statistics
  @staticmethod
  def __update_statistics(wall_time):
    SerialNBWorker.__shared_lock.acquire()
    try:
      SerialNBWorker.__total_encoding_and_decoding_time += wall_time
      SerialNBWorker.__total_number_of_noisy_blocks += 1
    except:
      logging.exception("{0} could not update its statistics!".format(self))
      raise
    finally:
      SerialNBWorker.__shared_lock.release()


  # thread-safe statistics
  @staticmethod
  def get_statistics():
    SerialNBWorker.__shared_lock.acquire()
    try:
      return(SerialNBWorker.__total_encoding_and_decoding_time,
              SerialNBWorker.__total_number_of_noisy_blocks)
    except:
      logging.exception("{0} could not read its statistics!".format(self))
      raise
    finally:
      SerialNBWorker.__shared_lock.release()


  def work(self):
    try:
      start_wall_time = timeit.default_timer()

      # Bitstring of N bits with exactly one bit turned on
      PS = libmultiblockpir.empty_array(self.N)
      PS[self.block_index] = 1

      # CR (1xN) += PS (1xN)
      libmultiblockpir.print_array(self.CR, self.N)
      libmultiblockpir.xor_vector(self.CR, PS, self.CR, self.N)
      libmultiblockpir.print_array(self.CR, self.N)

      stop_wall_time = timeit.default_timer()
      wall_time = stop_wall_time-start_wall_time

      # NB (1xS) = CR (1xN) x D (NxS)
      NB = self.mirror.multiply_and_add_with_database(self.W, self.CR, self.S)

      start_wall_time = timeit.default_timer()

      # Decode NB with GE and RR.
      # NB += GE[i] x RR[i] for i in {1..n}
      for i in xrange(self.n):
        self.multiply_and_add_vector_with_scalar(self.GE[i], self.S,
                                                 self.RR[i], NB)

      assert self.block_index not in self.fetched_blocks
      self.fetched_blocks[self.block_index] = NB

      stop_wall_time = timeit.default_timer()
      wall_time += stop_wall_time-start_wall_time
      SerialNBWorker.__update_statistics(wall_time)
    except:
      logging.exception("SerialNBWorker({0}) aborted!".format(
                        self.block_index))
      raise
    else:
      logging.info("SerialNBWorker({0}) passed".format(self.block_index))





class ParallelNBWorker(SerialNBWorker, threading.Thread):
  """A concurrent method to download a Noisy/New Block (NB)."""

  def __init__(self, CR, GE, N, RR, S, W, barrier, block_index, fetched_blocks,
               n, mirror, multiply_and_add_vector_with_scalar):
    threading.Thread.__init__(self)
    self.daemon = True

    SerialNBWorker.__init__(self, CR, GE, N, RR, S, W, block_index,
                            fetched_blocks, n, mirror,
                            multiply_and_add_vector_with_scalar)
    self.barrier = barrier


  def run(self):
    try:
      self.work()
    except:
      self.barrier.abort()
      logging.exception("ParallelNBWorker({0}) aborted the barrier!".format(
                        self.block_index))
    else:
      self.barrier.wait()
      logging.info("ParallelNBWorker({0}) passed the barrier".format(
                    self.block_index))





class KSafeBlockDownloader(object):
  """A convenient k-safe block downloader."""

  def __init__(self, mirror_controller, k, q, m=None):
    """Provide k-safety with GF(2^q) given m mirrors in mirror_controller.

    If m is None, we will default to mirror_controller.get_number_of_mirrors();
    otherwise, we will take the given value of m at face value."""

    # We must operate in either GF(2^1) or GF(2^8).
    assert q in (1, 8)
    W = 2**q

    self.F = py_ecc.ffield.FField(q)
    self.W = W

    # The GF(2) case.
    if W == 2:
      self.matrix_multiply = libmultiblockpir.matrix_multiply_in_GF2
      self.multiply_and_add_vector_with_scalar = \
        libmultiblockpir.multiply_and_add_vector_with_scalar_in_GF2
    # The GF(256) case.
    else:
      self.matrix_multiply = libmultiblockpir.matrix_multiply_in_GF256
      self.multiply_and_add_vector_with_scalar = \
        libmultiblockpir.multiply_and_add_vector_with_scalar_in_GF256

    m = m or mirror_controller.get_number_of_mirrors()

    # 0 < k < m
    assert k > 0
    # Only in the GF(256) case, we must assert that k < 256.
    if q == 8:
      assert k < W
    assert k < m

    self.k = k
    self.m = m

    logging.info("k, m = {0}, {1}".format(k, m))

    # A block ticket number must be in [0, m-n-1]; n will be determined later.
    self.block_ticket_number = 0
    # This is how we talk to known mirrors.
    self.mirror_controller = mirror_controller
    # This is how we safely serialize concurrent access to fetching blocks.
    self.fetch_blocks_lock = threading.Lock()

    # internal statistics
    self.__number_of_mixed_blocks = 0 # all MB ever generated
    self.__number_of_noisy_blocks = 0 # all NB ever generated
    self.__k_safe_matrices_generation_time = 0  # time to compute k-safe matrices
    self.__fetch_blocks_time = 0  # time spent fetching all NB blocks ever generated


  # TODO: Parallel, background, functional computation.
  def __generate_k_safe_matrices(self, N, S, concurrent=True,
                                  recycle_mirrors=False):
    """WARNING: If recycle_mirrors is True, we will sample mirrors with
    replacement if we find k or m lacking with respect to the observed number
    of mirrors. This will defeat k-safety and lead to direct loss of privacy!
    This option is in place only for internal evaluation of the protocol and
    will be removed later."""

    # We can reuse R and its associates over the lifetime of this run
    if not hasattr(self, 'R'):
      logging.debug("Generating R...")

      start_wall_time = timeit.default_timer()
      self.__generate_R(recycle_mirrors=recycle_mirrors)
      stop_wall_time = timeit.default_timer()

      wall_time = stop_wall_time-start_wall_time
      self.__k_safe_matrices_generation_time += wall_time
      logging.debug("...done.")

    logging.debug("Generating E, CL, CR...")

    start_wall_time = timeit.default_timer()
    self.__generate_E_CL_CR(N)
    stop_wall_time = timeit.default_timer()

    wall_time = stop_wall_time-start_wall_time
    self.__k_safe_matrices_generation_time += wall_time
    logging.debug("...done.")

    logging.debug("Generating MB, GE...")
    self.__generate_MB_GE(N, S, concurrent=concurrent,
                          recycle_mirrors=recycle_mirrors)
    logging.debug("...done.")


  def __generate_R(self, recycle_mirrors=False):
    """WARNING: If recycle_mirrors is True, we will sample mirrors with
    replacement if we find k or m lacking with respect to the observed number
    of mirrors. This will defeat k-safety and lead to direct loss of privacy!
    This option is in place only for internal evaluation of the protocol and
    will be removed later."""

    # W in (2^1, 2^8)
    assert self.W in (2, 256)

    # The GF(2) case.
    if self.W == 2:
      # This should return an m x n 2D array, where n = d(k, m).
      RT = libmultiblockpir.get_transposed_k_safe_binary_matrix(self.k, self.m)
      assert len(RT) == self.m

      # What is n = d(k, m)?
      n = len(RT[0])
      for row in RT:
        assert len(row) == n
    # The GF(256) case.
    else:
      # This should return an n x m 2D array, where n = k.
      RT = libmultiblockpir.get_k_safe_byte_matrix(self.k, self.m, self.F)
      assert len(RT) == self.k

      for row in RT:
        assert len(row) == self.m

      # n = k
      n = self.k

    # Now that we have confirmed at least a consistent 2D array pretending to
    # be matrix, we set n.
    logging.info("n = {0}".format(n))
    self.n = n

    # R is an n x m matrix.
    R = py_ecc.genericmatrix.GenericMatrix((self.n, self.m), 0, 1, self.F.Add,
                                           self.F.Subtract, self.F.Multiply,
                                           self.F.Divide)

    # The GF(2) case.
    if self.W == 2:
      # We set R = transpose(RT).
      for j in xrange(self.n):
        column = [RT[i][j] for i in xrange(self.m)]
        R.SetRow(j, column)
    # The GF(256) case.
    else:
      # We set R = RT.
      for i in xrange(self.n):
        row = RT[i]
        R.SetRow(i, row)

    # Left side of R is n x n.
    RL = R.SubMatrix(0, self.n-1 , 0, self.n-1)

    # Right side of R is n x (m-n).
    RR = R.SubMatrix(0, self.n-1, self.n, self.m-1)

    self.R = R                      # n x m
    self.RL = RL                    # n x n
    self.RL_inverse = RL.Inverse()  # n x n
    self.RR = RR                    # n x (m-n)

    # Finally, after determining n, we find (m-n) NB mirrors.
    self.NB_mirrors = \
      self.mirror_controller.get_NB_mirrors(self.m-self.n,
                                            recycle_mirrors=recycle_mirrors)


  def __generate_E_CL_CR(self, N):
    E = py_ecc.genericmatrix.GenericMatrix((N, self.n), 0, 1, self.F.Add,
                                    self.F.Subtract, self.F.Multiply,
                                    self.F.Divide)

    for i in xrange(N):
      arr = []
      for j in xrange(self.n):
        arr.append(random.randrange(self.W))  # append a random FF element
      E.SetRow(i, arr)

    self.E = E            # N x n
    self.CL = E * self.RL # N x n
    self.CR = E * self.RR # N x (m-n)


  def __generate_MB_GE(self, N, S, concurrent=True, recycle_mirrors=False):
    """WARNING: If recycle_mirrors is True, we will sample mirrors with
    replacement if we find k or m lacking with respect to the observed number
    of mirrors. This will defeat k-safety and lead to direct loss of privacy!
    This option is in place only for internal evaluation of the protocol and
    will be removed later.

    NOTE: We assume that the MB generation time is dominated by network costs,
    and so its computation time is negligible; therefore, we do not include it
    in the time to generate k-safe matrices."""

    # MB (nxS) = CL (nxN) x D (NxS)
    logging.debug("Generating MB...")

    # n "mixed" blocks from mirrors
    MB = libmultiblockpir.empty_array_pointer(self.n)
    n_mirrors = \
      self.mirror_controller.get_MB_mirrors(self.n,
                                            recycle_mirrors=recycle_mirrors)

    if concurrent is True:
      # We, too, must wait until we hear from all of the k mirrors.
      mb_barrier = compatibility.Barrier(self.n+1)

    for i in xrange(self.n):
      CL_i = self.get_CL_for_mirror(i, N)
      n_mirror = n_mirrors[i]

      if concurrent is True:
        mb_worker = ParallelMBWorker(CL_i, MB, S, self.W, mb_barrier, i,
                                     n_mirror)
        mb_worker.start()
      else:
        mb_worker = SerialMBWorker(CL_i, MB, S, self.W, i, n_mirror)
        mb_worker.work()

      # Increment count of all MB ever generated by 1
      self.__number_of_mixed_blocks += 1

    if concurrent is True:
      mb_barrier.wait()

    logging.debug("...done.")

    # GE (nxS) = inverse(RL) (nxn) x MB (nxS)
    logging.debug("Generating GE...")
    start_wall_time = timeit.default_timer()

    # a ctypes copy of self.RL_inverse
    RL_inverse = libmultiblockpir.empty_array_pointer(self.n)
    # n "building" blocks from mirrors
    GE = libmultiblockpir.empty_array_pointer(self.n)

    for i in xrange(self.n):
      # allocate ctype arrays
      RL_inverse[i] = libmultiblockpir.empty_array(self.n)
      GE[i] = libmultiblockpir.empty_array(S)

      for j in xrange(self.n):
        RL_inverse[i][j] = self.RL_inverse[j, i]  # transposition

    self.matrix_multiply(RL_inverse, MB, GE, self.n)
    stop_wall_time = timeit.default_timer()

    wall_time = stop_wall_time-start_wall_time
    self.__k_safe_matrices_generation_time += wall_time
    logging.debug("...done.")

    self.MB = MB  # n x S
    self.GE = GE  # n x S


  # FIXME: Better design!
  def fetch_blocks(self, block_indices, concurrent=True, recycle_mirrors=False):
    """Fetch blocks with block_indices; returns a dictionary of blocks indexed
    by block_indices.

    WARNING: If recycle_mirrors is True, we will sample mirrors with replacement
    if we find k or m lacking with respect to the observed number of mirrors.
    This will defeat k-safety and lead to direct loss of privacy! This option
    is in place only for internal evaluation of the protocol and will be
    removed later."""

    N = self.mirror_controller.N
    S = self.mirror_controller.S

    # Sanity checks
    block_indices = set(block_indices)
    for block_index in block_indices:
      assert block_index >= 0
      assert block_index < N
    # TODO: len(block_indices) must be reasonable,
    # or we must limit the number of threads running at any time

    # Fetched blocks are stored by block indices.
    fetched_blocks = {}

    if concurrent is True:
      # We, too, must wait until we hear from all of the mirrors.
      barrier = compatibility.Barrier(len(block_indices)+1)

    self.fetch_blocks_lock.acquire()

    try:
      start_wall_time = timeit.default_timer()

      # TODO: parallel block *processing* (besides downloading)
      for block_index in block_indices:
        logging.debug("Fetching block {0} with ticket #{1}".format(block_index,
                      self.block_ticket_number))

        # Do we need new k-safe matrices?
        if self.block_ticket_number == 0:
          logging.debug("Generating new {0}-safe matrices...".format(self.k))
          self.__generate_k_safe_matrices(N, S, concurrent=concurrent,
                                          recycle_mirrors=recycle_mirrors)
          logging.debug("...done.")

        CR = self.get_CR_for_mirror(N)
        RR = self.get_RR_for_mirror()
        GE = self.GE[:] # make a copy
        mirror = self.NB_mirrors[self.block_ticket_number]

        if concurrent is True:
          block_fetcher = \
            ParallelNBWorker(CR, GE, N, RR, S, self.W, barrier, block_index,
                             fetched_blocks, self.n, mirror,
                             self.multiply_and_add_vector_with_scalar)
          block_fetcher.start()
        else:
          block_fetcher = \
            SerialNBWorker(CR, GE, N, RR, S, self.W, block_index,
                           fetched_blocks, self.n, mirror,
                           self.multiply_and_add_vector_with_scalar)
          block_fetcher.work()

        # Increment counters
        self.__number_of_noisy_blocks += 1
        self.block_ticket_number = \
          (self.block_ticket_number+1) % (self.m-self.n)
    except:
      logging.exception("Failed to fetch blocks {0}!".format(block_indices))
    else:
      barrier.wait()
      stop_wall_time = timeit.default_timer()
      wall_time = stop_wall_time-start_wall_time
      self.__fetch_blocks_time += wall_time
    finally:
      self.fetch_blocks_lock.release()
      return fetched_blocks


  def get_CL_for_mirror(self, i, N):
    """Get the vector CL of N bytes for the mirror i of n."""

    # 0 <= i < n
    assert i >= 0
    assert i < self.n

    # 1 x N
    return (libmultiblockpir.array_of_bytes(N))(*self.CL.GetColumn(i))


  def get_CR_for_mirror(self, N):
    """Get the vector CR of N bytes for the mirror i of m-n."""

    # 0 <= i < m-n
    i = self.block_ticket_number
    assert i >= 0
    assert i < self.m - self.n

    # 1 x N
    return (libmultiblockpir.array_of_bytes(N))(*self.CR.GetColumn(i))


  def get_RL_inverse_for_mirror(self, i):
    """Get the vector RL^-1 of n bytes for the mirror i of n."""

    # 0 <= i < n
    assert i >= 0
    assert i < self.n

    # 1 x n
    return (libmultiblockpir.array_of_bytes(self.n))(*self.RL_inverse.GetColumn(i))


  def get_RR_for_mirror(self):
    """Get the vector RR of n bytes for the mirror i of m-n."""

    # 0 <= i < m-n
    i = self.block_ticket_number
    assert i >= 0
    assert i < self.m - self.n

    # 1 x n
    return (libmultiblockpir.array_of_bytes(self.n))(*self.RR.GetColumn(i))


  def print_statistics(self):
    MB_to_NB_ratio = self.__number_of_mixed_blocks/self.__number_of_noisy_blocks

    total_network_time_to_mix_blocks = 0
    total_number_of_mixed_blocks = 0

    total_network_time_to_noise_blocks = 0  # not a verb, but you get the idea
    total_number_of_noisy_blocks = 0

    for MB_mirror in self.mirror_controller.get_MB_mirrors():
      network_time_to_mix_blocks, number_of_mixed_blocks = \
        MB_mirror.get_statistics()
      total_network_time_to_mix_blocks += network_time_to_mix_blocks
      total_number_of_mixed_blocks += number_of_mixed_blocks

    assert total_number_of_mixed_blocks == self.__number_of_mixed_blocks

    for NB_mirror in self.mirror_controller.get_NB_mirrors():
      network_time_to_noise_blocks, number_of_noisy_blocks = \
        NB_mirror.get_statistics()
      total_network_time_to_noise_blocks += network_time_to_noise_blocks
      total_number_of_noisy_blocks += number_of_noisy_blocks

    assert total_number_of_noisy_blocks == self.__number_of_noisy_blocks

    total_encoding_and_decoding_time, total_number_of_noisy_blocks = \
      SerialNBWorker.get_statistics()
    assert total_number_of_noisy_blocks == self.__number_of_noisy_blocks

    # Average matrices computation time over # of repetitions
    average_matrices_computation_time = self.__k_safe_matrices_generation_time/\
                                        (self.__number_of_mixed_blocks/self.n)

    average_noisy_block_coding_time = total_encoding_and_decoding_time/\
                                      self.__number_of_noisy_blocks

    average_block_network_time = \
      (total_network_time_to_mix_blocks+total_network_time_to_noise_blocks)/\
      (self.__number_of_mixed_blocks+self.__number_of_noisy_blocks)

    # Average block fetch time over # of noisy blocks,
    # so it includes the mixed blocks overhead
    average_block_fetch_time = self.__fetch_blocks_time/\
                                self.__number_of_noisy_blocks

    logging.info("observed overhead = {0}".format(MB_to_NB_ratio))
    logging.info("avg k-safe matrices computation time = {0}".format(
                  average_matrices_computation_time))
    logging.info("avg data block coding time = {0}".format(
                  average_noisy_block_coding_time))
    logging.info("avg block network time = {0}".format(
                  average_block_network_time))
    logging.info("avg data block fetch time = {0}".format(
                  average_block_fetch_time))
