# Multi-block private information retrieval (MB-PIR)

Private Information Retrieval (PIR) allows users
to retrieve information from a database without revealing
which information in the database was queried. The traditional
information-theoretic PIR schemes utilize multiple servers to
download a single data block, thus incurring high communication
overhead and high computation burdens. In this paper,
we develop an information-theoretic multi-block PIR scheme
that significantly reduces client communication and computation
overheads by downloading multiple data blocks at a time. The
design of k-safe binary matrices insures the information will not
be revealed even if up to k servers collude. Our scheme has much
lower overhead than classic PIR schemes. The implementation of
fast XOR operations benefits both servers and clients in reducing
coding and decoding time. Our work demonstrates that multiblock
PIR scheme can be optimized to simultaneously achieve
low communication and computation overhead, comparable to
even non-PIR systems, while maintaining a high level of privacy.
