#ifndef TQ_RELOPTIONS_H
#define TQ_RELOPTIONS_H

#include "postgres.h"

typedef struct TqAmOptions
{
	int32		vl_len_;
	int			bits;
	int			lists;
	int			router_samples;
	int			router_iterations;
	int			router_seed;
	bool		normalized;
	int			transform_offset;
	int			lanes_offset;
} TqAmOptions;

extern bytea *tq_reloptions(Datum reloptions, bool validate);

#endif
