#ifndef TQ_PGVECTOR_COMPAT_H
#define TQ_PGVECTOR_COMPAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "postgres.h"

typedef struct Vector Vector;
typedef struct HalfVector HalfVector;

typedef enum TqVectorInputKind
{
	TQ_VECTOR_INPUT_VECTOR = 1,
	TQ_VECTOR_INPUT_HALFVEC = 2
} TqVectorInputKind;

extern bool tq_vector_copy_from_pgvector(const Vector *vector,
										 float *out,
										 size_t out_len,
										 uint32_t *dimension,
										 char *errmsg,
										 size_t errmsg_len);
extern bool tq_vector_copy_from_halfvec(const HalfVector *vector,
										float *out,
										size_t out_len,
										uint32_t *dimension,
										char *errmsg,
										size_t errmsg_len);
extern bool tq_vector_input_kind_from_typid(Oid type_oid,
											TqVectorInputKind *kind,
											char *errmsg,
											size_t errmsg_len);
extern bool tq_vector_dimension_from_datum(Datum value,
										   uint32_t *dimension,
										   char *errmsg,
										   size_t errmsg_len);
extern bool tq_vector_dimension_from_datum_typed(Datum value,
												 TqVectorInputKind kind,
												 uint32_t *dimension,
												 char *errmsg,
												 size_t errmsg_len);
extern bool tq_vector_copy_from_datum(Datum value,
									  float *out,
									  size_t out_len,
									  uint32_t *dimension,
									  char *errmsg,
									  size_t errmsg_len);
extern bool tq_vector_copy_from_datum_typed(Datum value,
											TqVectorInputKind kind,
											float *out,
											size_t out_len,
											uint32_t *dimension,
											char *errmsg,
											size_t errmsg_len);

#endif
