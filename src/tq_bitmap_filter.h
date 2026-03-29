#ifndef TQ_BITMAP_FILTER_H
#define TQ_BITMAP_FILTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "postgres.h"

#include "src/tq_page.h"
#include "src/tq_pgvector_compat.h"

extern bytea *tq_bitmap_filter_pack(TqDistanceKind distance_kind,
									const float *query_values,
									uint32_t dimension,
									double threshold);
extern bool tq_bitmap_filter_parse(bytea *filter,
								   TqDistanceKind expected_distance,
								   float **query_values,
								   uint32_t *dimension,
								   double *threshold,
								   char *errmsg,
								   size_t errmsg_len);
extern bool tq_bitmap_filter_match_datum(Datum value,
										 TqVectorInputKind input_kind,
										 bytea *filter,
										 TqDistanceKind expected_distance,
										 bool *matches,
										 char *errmsg,
										 size_t errmsg_len);

#endif
