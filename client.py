#!/usr/bin/env python


"""A client which simply downloads, with k-safety, blocks from m mirrors and
(optionally) checks them for correctness."""


# TODO: k, m should be command-line parameters.
# TODO: Compute n=d(k,m) instead of specifying it as a given parameter.


import logging
import random
import sys

import libmultiblockpir
import libclient





if __name__ == "__main__":
  line_format = "[%(asctime)s] [%(name)s] [%(levelname)s] [%(funcName)s:%(lineno)s@%(filename)s] %(message)s"
  logging.basicConfig(filename="client.log", format=line_format, level=logging.DEBUG)

  DEBUG = True
  # set AWS mirrors
  MIRRORS = set([
    "ec2-75-101-239-44.compute-1.amazonaws.com" # virginia
  ])

  # set parameters
  concurrent = True       # we want concurrent block requests
  port = 8000             # all experiment servers are on this port
  recycle_mirrors = True  # we are recyling mirrors for our experiments
  k = 7                   # need k+1 coalition members to break privacy
  m = 16                  # alleged number of mirrors
  n = 7                   # the number of rows in the matrix R
  q = 8                   # the finite field order
  counter = 64            # download up to #counter sets of blocks

  # sanity checks
  assert k <= n
  assert n < m

  # init mirror controller and block downloader
  mirror_controller = libclient.MirrorController()
  block_downloader = libclient.KSafeBlockDownloader(mirror_controller, k, q,
                                                    m=m)

  # register AWS mirrors
  for mirror in MIRRORS:
    mirror_controller.register_MB_mirror(mirror, port,
                                          recycle_mirrors=recycle_mirrors)
    mirror_controller.register_NB_mirror(mirror, port,
                                          recycle_mirrors=recycle_mirrors)

  assert mirror_controller.get_number_of_MB_mirrors() == len(MIRRORS)
  assert mirror_controller.get_number_of_NB_mirrors() == len(MIRRORS)

  # get N, S from mirror set
  N = mirror_controller.N
  S = mirror_controller.S

  if DEBUG:
    # Load database D into memory, and get number of blocks, N.
    # dd if=/dev/urandom of=almost.2GB.db bs=715827 count=3000
    # In that case, we are testing something just a little shy of 2GB.
    assert len(sys.argv) > 1
    filename = sys.argv[1]
    assert N == libmultiblockpir.memorize_database(filename)

  while counter > 0:
    # download m-n blocks at once
    assert m-n <= N
    block_indices = random.sample(xrange(N), m-n)
    assert len(set(block_indices)) == m-n

    fetched_blocks = \
      block_downloader.fetch_blocks(block_indices, concurrent=concurrent,
                                    recycle_mirrors=recycle_mirrors)
    counter -= 1

    if set(fetched_blocks.keys()) == set(block_indices):
      logging.info("Fetched {0} blocks".format(len(set(block_indices))))
    else:
      logging.warn("Failed to fetch {0} blocks".format(len(set(block_indices))))

    if DEBUG:
      original_block = libmultiblockpir.empty_array(S)

      for block_index in block_indices:
        fetched_block = fetched_blocks[block_index]
        # Fetch original block from database for comparison
        libmultiblockpir.block_at_index(block_index, original_block)

        # Is original block equal to decoded block?
        assert libmultiblockpir.are_equal_arrays(original_block, fetched_block, S)
        logging.info("Fetched block({0}) == original block({0})".format(block_index))

  # print statistics
  block_downloader.print_statistics()
