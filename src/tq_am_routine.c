#include "postgres.h"

#include "commands/vacuum.h"

#include "src/tq_am_routine.h"

void
tq_init_amroutine(IndexAmRoutine *amroutine)
{
	MemSet(amroutine, 0, sizeof(*amroutine));
	amroutine->type = T_IndexAmRoutine;

	amroutine->amstrategies = 1;
	amroutine->amsupport = 2;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
	amroutine->amcanbackward = false;
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = true;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = true;
	amroutine->amsearchnulls = true;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
	amroutine->amcaninclude = true;
	amroutine->amusemaintenanceworkmem = false;
	amroutine->amsummarizing = false;
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_NO_PARALLEL;
	amroutine->amkeytype = InvalidOid;
}
