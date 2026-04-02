#include "postgres.h"

#include <math.h>
#include <string.h>

#include "fmgr.h"
#include "varatt.h"
#include "utils/builtins.h"

#include "src/tq_bitmap_filter.h"

#define TQ_BITMAP_FILTER_MAGIC UINT32_C(0x54425146)
#define TQ_BITMAP_FILTER_VERSION UINT16_C(1)

PG_FUNCTION_INFO_V1(tq_bitmap_cosine_filter);
PG_FUNCTION_INFO_V1(tq_bitmap_cosine_filter_halfvec);
PG_FUNCTION_INFO_V1(tq_bitmap_cosine_match);
PG_FUNCTION_INFO_V1(tq_bitmap_cosine_match_halfvec);

static void
tq_bitmap_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg == NULL || errmsg_len == 0)
		return;

	strlcpy(errmsg, message, errmsg_len);
}

static bool
tq_bitmap_distance_from_vectors(TqDistanceKind distance_kind,
								const float *left,
								const float *right,
								uint32_t dimension,
								double *distance,
								char *errmsg,
								size_t errmsg_len)
{
	double		dot_product = 0.0;
	double		left_norm_squared = 0.0;
	double		right_norm_squared = 0.0;
	uint32_t	i = 0;

	if (left == NULL || right == NULL || distance == NULL)
	{
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: vectors and distance output must be non-null");
		return false;
	}

	for (i = 0; i < dimension; i++)
	{
		double	left_value = (double) left[i];
		double	right_value = (double) right[i];

		dot_product += left_value * right_value;
		left_norm_squared += left_value * left_value;
		right_norm_squared += right_value * right_value;
	}

	switch (distance_kind)
	{
		case TQ_DISTANCE_COSINE:
			if (left_norm_squared <= 0.0 || right_norm_squared <= 0.0)
			{
				*distance = 1.0;
				return true;
			}

			*distance = 1.0 - (dot_product / sqrt(left_norm_squared * right_norm_squared));
			if (*distance < 0.0 && *distance > -1e-12)
				*distance = 0.0;
			return true;
		case TQ_DISTANCE_IP:
			*distance = -dot_product;
			return true;
		case TQ_DISTANCE_L2:
			*distance = left_norm_squared + right_norm_squared - (2.0 * dot_product);
			if (*distance < 0.0 && *distance > -1e-12)
				*distance = 0.0;
			return true;
		default:
			tq_bitmap_set_error(errmsg, errmsg_len,
								"invalid turboquant bitmap filter: unsupported distance kind");
			return false;
	}
}

bytea *
tq_bitmap_filter_pack(TqDistanceKind distance_kind,
					  const float *query_values,
					  uint32_t dimension,
					  double threshold)
{
	uint32_t	magic = TQ_BITMAP_FILTER_MAGIC;
	uint16_t	version = TQ_BITMAP_FILTER_VERSION;
	uint16_t	stored_distance = (uint16_t) distance_kind;
	bytea	   *result;
	char	   *cursor;
	size_t		payload_bytes;

	payload_bytes = sizeof(uint32_t)
		+ sizeof(uint16_t)
		+ sizeof(uint16_t)
		+ sizeof(uint32_t)
		+ sizeof(double)
		+ (sizeof(float) * (size_t) dimension);

	result = (bytea *) palloc0(VARHDRSZ + payload_bytes);
	SET_VARSIZE(result, (int) (VARHDRSZ + payload_bytes));
	cursor = VARDATA(result);

	memcpy(cursor, &magic, sizeof(uint32_t));
	cursor += sizeof(uint32_t);
	memcpy(cursor, &version, sizeof(uint16_t));
	cursor += sizeof(uint16_t);
	memcpy(cursor, &stored_distance, sizeof(uint16_t));
	cursor += sizeof(uint16_t);
	memcpy(cursor, &dimension, sizeof(uint32_t));
	cursor += sizeof(uint32_t);
	memcpy(cursor, &threshold, sizeof(double));
	cursor += sizeof(double);
	if (dimension > 0)
		memcpy(cursor, query_values, sizeof(float) * (size_t) dimension);

	return result;
}

bool
tq_bitmap_filter_parse(bytea *filter,
					   TqDistanceKind expected_distance,
					   float **query_values,
					   uint32_t *dimension,
					   double *threshold,
					   char *errmsg,
					   size_t errmsg_len)
{
	char	   *cursor;
	size_t		payload_len;
	uint32_t	magic = 0;
	uint16_t	version = 0;
	uint16_t	distance_kind = 0;
	uint32_t	filter_dimension = 0;
	double		filter_threshold = 0.0;
	size_t		expected_len = 0;

	if (filter == NULL || query_values == NULL || dimension == NULL || threshold == NULL)
	{
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: filter outputs must be non-null");
		return false;
	}

	payload_len = (size_t) VARSIZE_ANY_EXHDR(filter);
	if (payload_len < sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint16_t)
		+ sizeof(uint32_t) + sizeof(double))
	{
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: payload is truncated");
		return false;
	}

	cursor = VARDATA_ANY(filter);
	memcpy(&magic, cursor, sizeof(uint32_t));
	cursor += sizeof(uint32_t);
	memcpy(&version, cursor, sizeof(uint16_t));
	cursor += sizeof(uint16_t);
	memcpy(&distance_kind, cursor, sizeof(uint16_t));
	cursor += sizeof(uint16_t);
	memcpy(&filter_dimension, cursor, sizeof(uint32_t));
	cursor += sizeof(uint32_t);
	memcpy(&filter_threshold, cursor, sizeof(double));
	cursor += sizeof(double);

	if (magic != TQ_BITMAP_FILTER_MAGIC)
	{
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: magic does not match");
		return false;
	}

	if (version != TQ_BITMAP_FILTER_VERSION)
	{
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: unsupported version");
		return false;
	}

	if ((TqDistanceKind) distance_kind != expected_distance)
	{
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: distance kind does not match operator");
		return false;
	}

	expected_len = sizeof(uint32_t)
		+ sizeof(uint16_t)
		+ sizeof(uint16_t)
		+ sizeof(uint32_t)
		+ sizeof(double)
		+ (sizeof(float) * (size_t) filter_dimension);
	if (payload_len != expected_len)
	{
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: payload length does not match dimension");
		return false;
	}

	if (filter_dimension == 0)
	{
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: query dimension must be positive");
		return false;
	}

	*query_values = (float *) palloc(sizeof(float) * (size_t) filter_dimension);
	memcpy(*query_values, cursor, sizeof(float) * (size_t) filter_dimension);
	*dimension = filter_dimension;
	*threshold = filter_threshold;
	return true;
}

bool
tq_bitmap_filter_match_datum(Datum value,
							 TqVectorInputKind input_kind,
							 bytea *filter,
							 TqDistanceKind expected_distance,
							 bool *matches,
							 char *errmsg,
							 size_t errmsg_len)
{
	float	   *left_values = NULL;
	float	   *right_values = NULL;
	uint32_t	left_dimension = 0;
	uint32_t	right_dimension = 0;
	double		threshold = 0.0;
	double		distance = 0.0;
	bool		ok = false;

	if (matches == NULL)
	{
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: match output must be non-null");
		return false;
	}

	if (!tq_bitmap_filter_parse(filter, expected_distance, &right_values,
								&right_dimension, &threshold,
								errmsg, errmsg_len))
		return false;

	left_values = (float *) palloc(sizeof(float) * (size_t) right_dimension);
	ok = tq_vector_copy_from_datum_typed(value, input_kind, left_values,
										 right_dimension, &left_dimension,
										 errmsg, errmsg_len);
	if (!ok)
	{
		pfree(left_values);
		pfree(right_values);
		return false;
	}

	if (left_dimension != right_dimension)
	{
		pfree(left_values);
		pfree(right_values);
		tq_bitmap_set_error(errmsg, errmsg_len,
							"invalid turboquant bitmap filter: vector dimension does not match filter");
		return false;
	}

	ok = tq_bitmap_distance_from_vectors(expected_distance,
										 left_values,
										 right_values,
										 left_dimension,
										 &distance,
										 errmsg,
										 errmsg_len);
	pfree(left_values);
	pfree(right_values);
	if (!ok)
		return false;

	*matches = distance <= threshold;
	return true;
}

Datum
tq_bitmap_cosine_filter(PG_FUNCTION_ARGS)
{
	Datum		value = PG_GETARG_DATUM(0);
	float	   *query_values = NULL;
	uint32_t	dimension = 0;
	bytea	   *result;
	char		error_buf[256];

	memset(error_buf, 0, sizeof(error_buf));
	if (!tq_vector_dimension_from_datum_typed(value, TQ_VECTOR_INPUT_VECTOR,
											  &dimension, error_buf, sizeof(error_buf)))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));

	query_values = (float *) palloc(sizeof(float) * (size_t) dimension);
	if (!tq_vector_copy_from_datum_typed(value, TQ_VECTOR_INPUT_VECTOR,
										 query_values, (size_t) dimension,
										 &dimension, error_buf, sizeof(error_buf)))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));

	result = tq_bitmap_filter_pack(TQ_DISTANCE_COSINE,
								   query_values,
								   dimension,
								   PG_GETARG_FLOAT8(1));
	pfree(query_values);
	PG_RETURN_BYTEA_P(result);
}

Datum
tq_bitmap_cosine_filter_halfvec(PG_FUNCTION_ARGS)
{
	Datum		value = PG_GETARG_DATUM(0);
	float	   *query_values = NULL;
	uint32_t	dimension = 0;
	bytea	   *result;
	char		error_buf[256];

	memset(error_buf, 0, sizeof(error_buf));
	if (!tq_vector_dimension_from_datum_typed(value, TQ_VECTOR_INPUT_HALFVEC,
											  &dimension, error_buf, sizeof(error_buf)))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));

	query_values = (float *) palloc(sizeof(float) * (size_t) dimension);
	if (!tq_vector_copy_from_datum_typed(value, TQ_VECTOR_INPUT_HALFVEC,
										 query_values, (size_t) dimension,
										 &dimension, error_buf, sizeof(error_buf)))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));

	result = tq_bitmap_filter_pack(TQ_DISTANCE_COSINE,
								   query_values,
								   dimension,
								   PG_GETARG_FLOAT8(1));
	pfree(query_values);
	PG_RETURN_BYTEA_P(result);
}

Datum
tq_bitmap_cosine_match(PG_FUNCTION_ARGS)
{
	bool		matches = false;
	char		error_buf[256];

	memset(error_buf, 0, sizeof(error_buf));
	if (!tq_bitmap_filter_match_datum(PG_GETARG_DATUM(0),
									  TQ_VECTOR_INPUT_VECTOR,
									  PG_GETARG_BYTEA_PP(1),
									  TQ_DISTANCE_COSINE,
									  &matches,
									  error_buf,
									  sizeof(error_buf)))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));

	PG_RETURN_BOOL(matches);
}

Datum
tq_bitmap_cosine_match_halfvec(PG_FUNCTION_ARGS)
{
	bool		matches = false;
	char		error_buf[256];

	memset(error_buf, 0, sizeof(error_buf));
	if (!tq_bitmap_filter_match_datum(PG_GETARG_DATUM(0),
									  TQ_VECTOR_INPUT_HALFVEC,
									  PG_GETARG_BYTEA_PP(1),
									  TQ_DISTANCE_COSINE,
									  &matches,
									  error_buf,
									  sizeof(error_buf)))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));

	PG_RETURN_BOOL(matches);
}
