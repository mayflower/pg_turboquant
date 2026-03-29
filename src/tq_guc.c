#include "postgres.h"

#include "utils/guc.h"

#include "src/tq_guc.h"
#include "src/tq_query_tuning.h"

int			tq_guc_probes = TQ_DEFAULT_PROBES;
int			tq_guc_oversample_factor = TQ_DEFAULT_OVERSAMPLE_FACTOR;

void		_PG_init(void);

void
_PG_init(void)
{
	DefineCustomIntVariable("turboquant.probes",
							"TurboQuant query breadth budget.",
							"Flat mode uses probes with oversample_factor to bound candidate retention; IVF mode will use it for list probes.",
							&tq_guc_probes,
							TQ_DEFAULT_PROBES,
							TQ_MIN_TUNING_VALUE,
							TQ_MAX_TUNING_VALUE,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.oversample_factor",
							"TurboQuant candidate oversampling multiplier.",
							"Controls how many approximate candidates are retained relative to the breadth budget.",
							&tq_guc_oversample_factor,
							TQ_DEFAULT_OVERSAMPLE_FACTOR,
							TQ_MIN_TUNING_VALUE,
							TQ_MAX_TUNING_VALUE,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);
}
