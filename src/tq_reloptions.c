#include "postgres.h"

#include <limits.h>

#include "access/reloptions.h"
#include "utils/builtins.h"

#include "src/tq_options.h"
#include "src/tq_reloptions.h"

static void tq_reloptions_validator(void *parsed_options, relopt_value *vals, int nvals);
static void tq_init_local_relopts(local_relopts *relopts);
static Size tq_fill_string_relopt_value(const char *value, void *ptr);

static void
tq_reloptions_validator(void *parsed_options, relopt_value *vals, int nvals)
{
	TqAmOptions *options = (TqAmOptions *) parsed_options;
	TqOptionConfig config = {
		.bits = options->bits,
		.lists = options->lists,
		.router_samples = options->router_samples,
		.router_iterations = options->router_iterations,
		.router_restarts = options->router_restarts,
		.router_seed = options->router_seed,
		.qjl_sketch_dim = options->qjl_sketch_dim,
		.normalized = options->normalized,
		.transform_name = GET_STRING_RELOPTION(options, transform_offset),
		.lanes_name = GET_STRING_RELOPTION(options, lanes_offset)
	};
	char		error_message[256];

	(void) vals;
	(void) nvals;

	if (!tq_validate_option_config(&config, error_message, sizeof(error_message)))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("%s", error_message)));
}

static void
tq_init_local_relopts(local_relopts *relopts)
{
	init_local_reloptions(relopts, sizeof(TqAmOptions));
	add_local_int_reloption(relopts, "bits",
							"TurboQuant code width in bits",
							4, 2, 8,
							offsetof(TqAmOptions, bits));
	add_local_int_reloption(relopts, "lists",
							"TurboQuant list count; 0 selects flat mode",
							0, 0, INT_MAX,
							offsetof(TqAmOptions, lists));
	add_local_int_reloption(relopts, "router_samples",
							"TurboQuant IVF router training sample count",
							256, 1, 65536,
							offsetof(TqAmOptions, router_samples));
	add_local_int_reloption(relopts, "router_iterations",
							"TurboQuant IVF router Lloyd iteration budget",
							8, 1, 64,
							offsetof(TqAmOptions, router_iterations));
	add_local_int_reloption(relopts, "router_restarts",
							"TurboQuant IVF router deterministic restart budget",
							3, 1, 8,
							offsetof(TqAmOptions, router_restarts));
	add_local_int_reloption(relopts, "router_seed",
							"TurboQuant IVF router deterministic seed",
							20260327, 0, INT_MAX,
							offsetof(TqAmOptions, router_seed));
	add_local_int_reloption(relopts, "qjl_sketch_dim",
							"TurboQuant residual QJL projection dimension; 0 selects the transformed dimension",
							0, 0, 65536,
							offsetof(TqAmOptions, qjl_sketch_dim));
	add_local_bool_reloption(relopts, "normalized",
							 "Whether input vectors are already normalized",
							 true,
							 offsetof(TqAmOptions, normalized));
	add_local_string_reloption(relopts, "transform",
							   "Structured transform family",
							   "hadamard",
							   NULL, tq_fill_string_relopt_value,
							   offsetof(TqAmOptions, transform_offset));
	add_local_string_reloption(relopts, "lanes",
							   "Lane count selection policy",
							   "auto",
							   NULL, tq_fill_string_relopt_value,
							   offsetof(TqAmOptions, lanes_offset));
	register_reloptions_validator(relopts, tq_reloptions_validator);
}

static Size
tq_fill_string_relopt_value(const char *value, void *ptr)
{
	Size		value_len = 0;

	if (value == NULL)
		return 0;

	value_len = strlen(value) + 1;
	if (ptr != NULL)
		memcpy(ptr, value, value_len);
	return value_len;
}

bytea *
tq_reloptions(Datum reloptions, bool validate)
{
	local_relopts relopts;

	tq_init_local_relopts(&relopts);
	return (bytea *) build_local_reloptions(&relopts,
											reloptions,
											validate);
}
