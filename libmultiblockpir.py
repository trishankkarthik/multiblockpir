import ctypes
import math
import logging
import random


################################ GLOBAL CLASSES ################################


class InvalidKSafeMatrixError(Exception):
  """Raised when the given (k, m) parameters do not lead to a proper k-safe
  matrix R."""

  pass


############################### GLOBAL VARIABLES ###############################


libmultiblockpir_c = ctypes.cdll.LoadLibrary('./libmultiblockpir.so.1')


# Toggle this to execute certain code in debug mode
DEBUG = False


# Block size
S = ctypes.c_int.in_dll(libmultiblockpir_c, 'S').value


# void block_at_index(const size_t index, byte* const block)
# What is the block at this index?
block_at_index = libmultiblockpir_c.block_at_index


# size_t memorize_database(const char* filename)
# Read data from filename into database D, and return N (number of blocks).
_memorize_database = libmultiblockpir_c.memorize_database
_memorize_database.restype = ctypes.c_size_t


# void matrix_multiply_in_GF2(const byte** const RL_inverse,
#                             const byte** const MB, byte** GE, size_t k);
# void matrix_multiply_in_GF256(const byte** const RL_inverse,
#                               const byte** const MB, byte** GE, size_t k);

matrix_multiply_in_GF2 = libmultiblockpir_c.matrix_multiply_in_GF2
matrix_multiply_in_GF256 = libmultiblockpir_c.matrix_multiply_in_GF256

matrix_multiply_in_GF2.argtypes = matrix_multiply_in_GF256.argtypes = [
  ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
  ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
  ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.c_size_t]


# B = D * E

# void multiply_and_add_with_database_in_GF2(const byte* const E,
#                                            byte* const B)
# void multiply_and_add_with_database_in_GF256(const byte* const E,
#                                              byte* const B)
multiply_and_add_with_database_in_GF2 = \
  libmultiblockpir_c.multiply_and_add_with_database_in_GF2

multiply_and_add_with_database_in_GF256 = \
  libmultiblockpir_c.multiply_and_add_with_database_in_GF256

multiply_and_add_with_database_in_GF2.argtypes = \
  multiply_and_add_with_database_in_GF256.argtypes = \
    [ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte)]


# sum_vector += vector * scalar

# void multiply_and_add_vector_with_scalar_in_GF2(const byte* const vector,
#                                                 const size_t vector_length,
#                                                 const byte scalar,
#                                                 byte* const sum_vector)
# void multiply_and_add_vector_with_scalar_in_GF256(const byte* const vector,
#                                                   const size_t vector_length,
#                                                   const byte scalar,
#                                                   byte* const sum_vector)
multiply_and_add_vector_with_scalar_in_GF2 = \
  libmultiblockpir_c.multiply_and_add_vector_with_scalar_in_GF2

multiply_and_add_vector_with_scalar_in_GF256 = \
  libmultiblockpir_c.multiply_and_add_vector_with_scalar_in_GF256


# sum_vector = vector1 + vector2

# void xor_vector(const char* const vector1, const char* const vector2,
#                 char* const sum_vector, const size_t vector_length);
xor_vector = libmultiblockpir_c.xor_vector


############################### GLOBAL FUNCTIONS ###############################


def array_of_bytes(N):
  return ctypes.c_ubyte*N


def are_equal_arrays(array1, array2, N):
  """Is array1 == array2?"""

  logging.debug("N == {0}".format(N))
  logging.debug("len(array1) == {0}".format(len(array1)))
  logging.debug("len(array2) == {0}".format(len(array2)))
  logging.debug("sum(array1) == {0}".format(sum(array1)))
  logging.debug("sum(array2) == {0}".format(sum(array2)))

  assert len(array1) == N
  assert len(array2) == N

  equal = True

  for i in xrange(N):
    equal = equal and (array1[i] == array2[i])

  return equal


def bzero(array, N):
  """Zero out array."""

  for i in xrange(N):
    array[i] = 0


def block_index(array, N):
  """What is the desired block index in the position vector array?"""

  index = None

  for i in xrange(N):
    element = array[i]
    if element == 1:
      assert index is None
      index = i
    else:
      assert element == 0

  return index


def empty_array(N):
  """Return an empty array of N bytes, each initialized to zero."""

  return (ctypes.c_ubyte*N)() # array of N bytes initialized to zeroes


def empty_array_pointer(N):
  """Return array of N arrays of bytes."""

  return (ctypes.POINTER(ctypes.c_ubyte)*N)() # array of N arrays of bytes


def expand_k_safe(k, m):
  """Luke's code."""

  k = int(k)
  subk = math.floor(k / 2)

  if k == 1:
    arr = []
    for i in range(m):
      arr.append([1])
    return arr
  else: # k != 1
    rnd = get_round(k, m)
    R_tl = get_iden(k)
    for i in range(rnd):
      R_tr = list(R_tl)
      R_br = expand_k_safe(subk, (k + 1) * (2 ** i))
      dim = len(R_br[0])
      R_bl = []
      for j in range((k + 1) * (2 ** i)):
        R_bl.append([0] * dim)
      for j in range(len(R_tl)):
        R_tl[j] = R_tl[j] + R_bl[j]
      for j in range(len(R_tr)):
        R_tr[j] = R_tr[j] + R_br[j]
      R_tl = R_tl + R_tr
    return R_tl


# a binary k-safe matrix has dimension d(k, m) x m, we need to know d(k, m)
def get_dimension(ksafematrix):
  """Luke's code."""

  # NOTE: Why does this return m and not d(k, m)?
  return len(ksafematrix[0])


def get_iden(k):
  """Luke's code."""

  #construct R_tl
  R_tl = []
  for i in range(k): # create the ith column vector
    tmparr = [0] * k
    tmparr[i] = 1
    R_tl.append(tmparr)
  tmparr = [1] * k
  R_tl.append(tmparr)
  return R_tl


def get_k_safe_byte_matrix(k, m, F):
  """Luke's code.

  k is the value in k-safety.
  m is the total number of mirrors.
  F is the finite field."""

  # Temporary, intermediate computation.
  arr = [[] for _ in xrange(k)]
  arr[0] = [1] * m

  tmparr = []
  for i in xrange(m):
    tmparr.append(i)
  arr[1] = tmparr

  for i in xrange(k-2):
    tmparr = []
    for j in xrange(m):
      tmparr.append(F.Multiply(arr[i+1][j], arr[1][j]))
    arr[i+2] = tmparr

  return arr


def get_order(ksafematrix, m):
  """Luke's code."""

  dim = get_dimension(ksafematrix)
  #print 'Dimension:', dim
  for i in range(dim):
    #print 'check vector ' + str(i + 1) + ' :', ksafematrix[i]
    if ksafematrix[i][i] != 1:
      #print str(i + 1) + 'th vector is invalid'
      for j in range(i, m):
        #print i, ksafematrix[j], ksafematrix[j][i]
        if ksafematrix[j][i] == 1:
          #print 'replacing ' + str(i + 1) + 'th vector with ' + str(j + 1) + 'th vector:', ksafematrix[j]
          tmpvec = ksafematrix[i]
          ksafematrix[i] = ksafematrix[j]
          ksafematrix[j] = tmpvec
          break
      #print 'New k-safe matrix:'
      #print ksafematrix
  return ksafematrix


def get_round(k, m):
  """Luke's code."""

  i = 0
  while True:
    if 2 ** i * (k + 1)  >= m:
      # NOTE: Whatever happens to rnd?
      rnd = i
      break
    else:
      i += 1
  return i


def get_transposed_k_safe_binary_matrix(k, m):
  """Luke's code.

  This is used to generate a k-safe matrix with m vectors.
  The dimension of this matrix is d(k, m) x m.
  This is different from the finite field case which gives dimension k x m.

  NOTE: Why does this return the matrix in m x d(k, m) dimensions?"""

  # NOTE: Why [:m]?
  ksafematrix = expand_k_safe(k, m)[:m]

  # But the matrix is in transpose...so is this correct?
  if get_dimension(ksafematrix) >= m:
    raise InvalidKSafeMatrixError()
  else:
      return get_order(ksafematrix, m)


def memorize_database(filename):
  """Read fileinto into database D."""

  return _memorize_database(filename)


def random_block_request(array, N):
  """Return a random position vector array."""

  random_block = random.randint(1, N)
  array[random_block-1] = 1


def print_array(array, N):
  """Print the given array."""

  if DEBUG:
    for i in xrange(N):
      print(array[i]),
    print('')
