#include "src/tq_options.h"

#include <stdio.h>
#include <string.h>

#ifdef TQ_UNIT_TEST
#undef snprintf
#endif

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static size_t
tq_div_ceil(size_t numerator, size_t denominator)
{
	return (numerator + denominator - 1) / denominator;
}

bool
tq_parse_transform_name(const char *transform_name,
						 TqTransformKind *transform_kind,
						 char *errmsg,
						 size_t errmsg_len)
{
	if (transform_name == NULL || transform_name[0] == '\0')
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"transform\": turboquant transform must be \"hadamard\" in v1");
		return false;
	}

	if (strcmp(transform_name, "hadamard") != 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"transform\": turboquant transform must be \"hadamard\" in v1");
		return false;
	}

	if (transform_kind != NULL)
		*transform_kind = TQ_TRANSFORM_HADAMARD;

	return true;
}

bool
tq_validate_lanes_name(const char *lanes_name, char *errmsg, size_t errmsg_len)
{
	if (lanes_name == NULL || lanes_name[0] == '\0')
		return true;

	if (strcmp(lanes_name, "auto") != 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"lanes\": turboquant lanes must be set to auto in v1");
		return false;
	}

	return true;
}

bool
tq_validate_option_config(const TqOptionConfig *config, char *errmsg, size_t errmsg_len)
{
	int			router_samples = config->router_samples == 0 ? 256 : config->router_samples;
	int			router_iterations = config->router_iterations == 0 ? 8 : config->router_iterations;
	int			router_restarts = config->router_restarts == 0 ? 3 : config->router_restarts;
	int			qjl_sketch_dim = config->qjl_sketch_dim;

	if (config->bits < 2 || config->bits > 8)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"bits\": turboquant bits must be between 2 and 8");
		return false;
	}

	if (config->lists < 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"lists\": turboquant lists must be greater than or equal to 0");
		return false;
	}

	if (router_samples < 1 || router_samples > 65536)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"router_samples\": turboquant router_samples must be between 1 and 65536");
		return false;
	}

	if (router_iterations < 1 || router_iterations > 64)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"router_iterations\": turboquant router_iterations must be between 1 and 64");
		return false;
	}

	if (router_restarts < 1 || router_restarts > 8)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"router_restarts\": turboquant router_restarts must be between 1 and 8");
		return false;
	}

	if (config->router_seed < 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"router_seed\": turboquant router_seed must be greater than or equal to 0");
		return false;
	}

	if (qjl_sketch_dim < 0 || qjl_sketch_dim > 65536)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"qjl_sketch_dim\": turboquant qjl_sketch_dim must be between 0 and 65536");
		return false;
	}

	if (!tq_parse_transform_name(config->transform_name, NULL, errmsg, errmsg_len))
		return false;

	if (!tq_validate_lanes_name(config->lanes_name, errmsg, errmsg_len))
		return false;

	return true;
}

bool
tq_compute_code_bytes(const TqLaneConfig *config,
					   size_t *code_bytes,
					   char *errmsg,
					   size_t errmsg_len)
{
	size_t		bytes = 0;
	int			qjl_dimension = config->qjl_dimension == 0 ? config->dimension : config->qjl_dimension;

	if (config->block_size <= 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid page budget: block size must be positive");
		return false;
	}

	if (config->dimension <= 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid page budget: dimension must be positive");
		return false;
	}

	if (qjl_dimension <= 0 || qjl_dimension > config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid page budget: qjl sketch dimension must be between 1 and the transformed dimension");
		return false;
	}

	if (config->bits < 2 || config->bits > 8)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid value for parameter \"bits\": turboquant bits must be between 2 and 8");
		return false;
	}

	if (config->tid_bytes <= 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid page budget: tid bytes must be positive");
		return false;
	}

	switch (config->codec)
	{
		case TQ_CODEC_PROD:
			bytes += tq_div_ceil((size_t) config->dimension * (size_t) (config->bits - 1), 8);
			bytes += tq_div_ceil((size_t) qjl_dimension, 8);
			bytes += sizeof(float);
			break;
		case TQ_CODEC_MSE:
			bytes += tq_div_ceil((size_t) config->dimension * (size_t) config->bits, 8);
			break;
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid page budget: unsupported codec");
			return false;
	}

	bytes += (size_t) config->tid_bytes;

	*code_bytes = bytes;
	return true;
}

bool
tq_resolve_lane_count(const TqLaneConfig *config,
					   int *lane_count,
					   char *errmsg,
					   size_t errmsg_len)
{
	static const int supported_lanes[] = {16, 8, 4, 2, 1};
	size_t		code_bytes = 0;
	int			usable_page_bytes = 0;
	size_t		i;

	if (!tq_compute_code_bytes(config, &code_bytes, errmsg, errmsg_len))
		return false;

	usable_page_bytes = config->block_size
		- config->page_header_bytes
		- config->special_space_bytes
		- config->reserve_bytes;

	if (usable_page_bytes <= 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid page budget: usable page bytes must be positive");
		return false;
	}

	for (i = 0; i < sizeof(supported_lanes) / sizeof(supported_lanes[0]); i++)
	{
		if ((size_t) supported_lanes[i] * code_bytes <= (size_t) usable_page_bytes)
		{
			*lane_count = supported_lanes[i];
			return true;
		}
	}

	tq_set_error(errmsg, errmsg_len,
				 "invalid page budget: one turboquant code does not fit on a page with the current settings");
	return false;
}
