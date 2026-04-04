#include "src/tq_pgvector_compat.h"

#include <stdlib.h>
#include <string.h>

#ifdef TQ_UNIT_TEST
#define tq_pgvector_detoast(value) ((Vector *) DatumGetPointer(value))
#define tq_halfvec_detoast(value) ((HalfVector *) DatumGetPointer(value))
#define tq_pgvector_release(vector, original) ((void) (vector), (void) (original))
#define tq_halfvec_release(vector, original) ((void) (vector), (void) (original))
#else
#include "fmgr.h"
#include "catalog/namespace.h"
#include "varatt.h"
#define tq_pgvector_detoast(value) DatumGetVector(value)
#define tq_halfvec_detoast(value) DatumGetHalfVector(value)
#define tq_pgvector_release(vector, original) \
	do { \
		if ((Pointer) (vector) != (original)) \
			pfree(vector); \
	} while (0)
#define tq_halfvec_release(vector, original) \
	do { \
		if ((Pointer) (vector) != (original)) \
			pfree(vector); \
	} while (0)
#endif

#include "third_party/pgvector/src/halfutils.h"
#include "third_party/pgvector/src/halfvec.h"
#include "third_party/pgvector/src/vector.h"

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	size_t		message_len = 0;

	if (errmsg == NULL || errmsg_len == 0)
		return;

	message_len = strlen(message);
	if (message_len >= errmsg_len)
		message_len = errmsg_len - 1;

	memcpy(errmsg, message, message_len);
	errmsg[message_len] = '\0';
}

bool
tq_vector_copy_from_pgvector(const Vector *vector,
							 float *out,
							 size_t out_len,
							 uint32_t *dimension,
							 char *errmsg,
							 size_t errmsg_len)
{
	uint32_t	i = 0;

	if (vector == NULL || out == NULL || dimension == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: vector, output, and dimension must be non-null");
		return false;
	}

	if (vector->dim <= 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: vector dimension must be positive");
		return false;
	}

	if (out_len < (size_t) vector->dim)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: output buffer is too small");
		return false;
	}

	for (i = 0; i < (uint32_t) vector->dim; i++)
		out[i] = vector->x[i];

	*dimension = (uint32_t) vector->dim;
	return true;
}

bool
tq_vector_copy_from_halfvec(const HalfVector *vector,
							float *out,
							size_t out_len,
							uint32_t *dimension,
							char *errmsg,
							size_t errmsg_len)
{
	uint32_t	i = 0;

	if (vector == NULL || out == NULL || dimension == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: halfvec, output, and dimension must be non-null");
		return false;
	}

	if (vector->dim <= 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: halfvec dimension must be positive");
		return false;
	}

	if (out_len < (size_t) vector->dim)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: output buffer is too small");
		return false;
	}

	for (i = 0; i < (uint32_t) vector->dim; i++)
		out[i] = HalfToFloat4(vector->x[i]);

	*dimension = (uint32_t) vector->dim;
	return true;
}

size_t
tq_vector_storage_size(TqVectorInputKind kind, uint32_t dimension)
{
	switch (kind)
	{
		case TQ_VECTOR_INPUT_VECTOR:
			return VECTOR_SIZE(dimension);
		case TQ_VECTOR_INPUT_HALFVEC:
			return HALFVEC_SIZE(dimension);
		default:
			return 0;
	}
}

bool
tq_vector_input_kind_from_typid(Oid type_oid,
								TqVectorInputKind *kind,
								char *errmsg,
								size_t errmsg_len)
{
#ifdef TQ_UNIT_TEST
	(void) type_oid;
	(void) kind;
	tq_set_error(errmsg, errmsg_len,
				 "invalid tq_pgvector conversion: type lookup is not available in unit tests");
	return false;
#else
	Oid			vector_typid = TypenameGetTypid("vector");
	Oid			halfvec_typid = TypenameGetTypid("halfvec");

	if (kind == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: type kind output must be non-null");
		return false;
	}

	if (type_oid == vector_typid && OidIsValid(vector_typid))
	{
		*kind = TQ_VECTOR_INPUT_VECTOR;
		return true;
	}

	if (type_oid == halfvec_typid && OidIsValid(halfvec_typid))
	{
		*kind = TQ_VECTOR_INPUT_HALFVEC;
		return true;
	}

	tq_set_error(errmsg, errmsg_len,
				 "invalid tq_pgvector conversion: unsupported operator class input type");
	return false;
#endif
}

bool
tq_vector_copy_from_datum_typed(Datum value,
								TqVectorInputKind kind,
								float *out,
								size_t out_len,
								uint32_t *dimension,
								char *errmsg,
								size_t errmsg_len)
{
	Pointer		original = DatumGetPointer(value);
	bool		ok = false;
	Vector	   *vector = NULL;
	HalfVector *halfvec = NULL;

	switch (kind)
	{
		case TQ_VECTOR_INPUT_VECTOR:
			vector = tq_pgvector_detoast(value);
			ok = tq_vector_copy_from_pgvector(vector, out, out_len, dimension,
											  errmsg, errmsg_len);
			tq_pgvector_release(vector, original);
			return ok;
		case TQ_VECTOR_INPUT_HALFVEC:
			halfvec = tq_halfvec_detoast(value);
			ok = tq_vector_copy_from_halfvec(halfvec, out, out_len, dimension,
											 errmsg, errmsg_len);
			tq_halfvec_release(halfvec, original);
			return ok;
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid tq_pgvector conversion: unsupported input kind");
			return false;
	}
}

bool
tq_vector_copy_raw_datum_typed(Datum value,
							   TqVectorInputKind kind,
							   uint8_t *out,
							   size_t out_len,
							   uint32_t *dimension,
							   char *errmsg,
							   size_t errmsg_len)
{
	Pointer		original = DatumGetPointer(value);
	Vector	   *vector = NULL;
	HalfVector *halfvec = NULL;
	size_t		raw_len = 0;

	if (out == NULL || dimension == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: raw output and dimension must be non-null");
		return false;
	}

	switch (kind)
	{
		case TQ_VECTOR_INPUT_VECTOR:
			vector = tq_pgvector_detoast(value);
			if (vector == NULL || vector->dim <= 0)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid tq_pgvector conversion: vector must be non-null with positive dimension");
				tq_pgvector_release(vector, original);
				return false;
			}
			raw_len = VECTOR_SIZE(vector->dim);
			if (out_len < raw_len)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid tq_pgvector conversion: raw vector output buffer is too small");
				tq_pgvector_release(vector, original);
				return false;
			}
			memcpy(out, vector, raw_len);
			*dimension = (uint32_t) vector->dim;
			tq_pgvector_release(vector, original);
			return true;
		case TQ_VECTOR_INPUT_HALFVEC:
			halfvec = tq_halfvec_detoast(value);
			if (halfvec == NULL || halfvec->dim <= 0)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid tq_pgvector conversion: halfvec must be non-null with positive dimension");
				tq_halfvec_release(halfvec, original);
				return false;
			}
			raw_len = HALFVEC_SIZE(halfvec->dim);
			if (out_len < raw_len)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid tq_pgvector conversion: raw halfvec output buffer is too small");
				tq_halfvec_release(halfvec, original);
				return false;
			}
			memcpy(out, halfvec, raw_len);
			*dimension = (uint32_t) halfvec->dim;
			tq_halfvec_release(halfvec, original);
			return true;
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid tq_pgvector conversion: unsupported input kind");
			return false;
	}
}

bool
tq_vector_datum_from_raw_bytes_typed(const uint8_t *raw_bytes,
									 size_t raw_len,
									 TqVectorInputKind kind,
									 uint32_t dimension,
									 Datum *value,
									 char *errmsg,
									 size_t errmsg_len)
{
	size_t expected_len = tq_vector_storage_size(kind, dimension);
	void   *copy = NULL;

	if (raw_bytes == NULL || value == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: raw input and datum output must be non-null");
		return false;
	}

	if (expected_len == 0 || raw_len != expected_len)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: raw value length does not match vector contract");
		return false;
	}

#ifdef TQ_UNIT_TEST
	copy = malloc(raw_len);
#else
	copy = palloc(raw_len);
#endif
	if (copy == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: out of memory constructing vector datum");
		return false;
	}
	memcpy(copy, raw_bytes, raw_len);
	*value = PointerGetDatum(copy);
	return true;
}

bool
tq_vector_copy_from_datum(Datum value,
						  float *out,
						  size_t out_len,
						  uint32_t *dimension,
						  char *errmsg,
						  size_t errmsg_len)
{
	return tq_vector_copy_from_datum_typed(value, TQ_VECTOR_INPUT_VECTOR,
										   out, out_len, dimension,
										   errmsg, errmsg_len);
}

bool
tq_vector_dimension_from_datum_typed(Datum value,
									 TqVectorInputKind kind,
									 uint32_t *dimension,
									 char *errmsg,
									 size_t errmsg_len)
{
	Pointer		original = DatumGetPointer(value);
	bool		ok = true;
	Vector	   *vector = NULL;
	HalfVector *halfvec = NULL;

	if (dimension == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_pgvector conversion: dimension output must be non-null");
		return false;
	}

	switch (kind)
	{
		case TQ_VECTOR_INPUT_VECTOR:
			vector = tq_pgvector_detoast(value);
			if (vector == NULL)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid tq_pgvector conversion: vector and dimension must be non-null");
				ok = false;
			}
			else if (vector->dim <= 0)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid tq_pgvector conversion: vector dimension must be positive");
				ok = false;
			}
			else
				*dimension = (uint32_t) vector->dim;
			tq_pgvector_release(vector, original);
			return ok;
		case TQ_VECTOR_INPUT_HALFVEC:
			halfvec = tq_halfvec_detoast(value);
			if (halfvec == NULL)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid tq_pgvector conversion: halfvec and dimension must be non-null");
				ok = false;
			}
			else if (halfvec->dim <= 0)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid tq_pgvector conversion: halfvec dimension must be positive");
				ok = false;
			}
			else
				*dimension = (uint32_t) halfvec->dim;
			tq_halfvec_release(halfvec, original);
			return ok;
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid tq_pgvector conversion: unsupported input kind");
			return false;
	}
}

bool
tq_vector_dimension_from_datum(Datum value,
							   uint32_t *dimension,
							   char *errmsg,
							   size_t errmsg_len)
{
	return tq_vector_dimension_from_datum_typed(value, TQ_VECTOR_INPUT_VECTOR,
												dimension, errmsg, errmsg_len);
}
