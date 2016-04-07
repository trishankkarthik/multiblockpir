// TYPE DEFINITIONS
typedef unsigned char byte;


// EXTERNAL VARIABLES
extern byte* D;   // database
extern size_t N;  // min # of blocks for database


// EXTERNAL CONSTANTS
extern const size_t S;          // How many bytes in a block?
extern const byte P[256][256];  // GF(256) multiplication lookup table


// FINITE-FIELD-INDEPENDENT FUNCTIONS
void block_at_index(const size_t index, byte* const block);

size_t memorize_database(const char* filename);

void xor_vector(const char* const vector1, const char* const vector2,
                char* const sum_vector, const size_t vector_length);


// GF(2) FUNCTIONS
void matrix_multiply_in_GF2(const byte** const RL_inverse,
                            const byte** const MB, byte** GE, size_t k);

void multiply_and_add_vector_with_scalar_in_GF2(const byte* const vector,
                                                const size_t vector_length,
                                                const byte scalar,
                                                byte* const sum_vector);

void multiply_and_add_with_database_in_GF2(const byte* const E, byte* const B);


// GF(256) FUNCTIONS
void matrix_multiply_in_GF256(const byte** const RL_inverse,
                              const byte** const MB, byte** GE, size_t k);

void multiply_and_add_vector_with_scalar_in_GF256(const byte* const vector,
                                                  const size_t vector_length,
                                                  const byte scalar,
                                                  byte* const sum_vector);

void multiply_and_add_with_database_in_GF256(const byte* const E,
                                             byte* const B);
