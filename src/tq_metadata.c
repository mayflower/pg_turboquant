#include "src/tq_metadata.h"

#include <stdio.h>
#include <string.h>

#include "catalog/pg_type_d.h"
#include "utils/uuid.h"

#ifdef TQ_UNIT_TEST
#undef snprintf
#define palloc0(sz) calloc(1, (sz))
#endif

static void
tq_metadata_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg_len == 0)
		return;
	snprintf(errmsg, errmsg_len, "%s", message);
}

bool
tq_metadata_kind_from_typid(Oid typid,
							  TqMetadataKind *kind,
							  char *errmsg,
							  size_t errmsg_len)
{
	TqMetadataKind resolved = TQ_METADATA_KIND_INVALID;

	switch (typid)
	{
		case BOOLOID:
			resolved = TQ_METADATA_KIND_BOOL;
			break;
		case INT2OID:
			resolved = TQ_METADATA_KIND_INT2;
			break;
		case INT4OID:
			resolved = TQ_METADATA_KIND_INT4;
			break;
		case INT8OID:
			resolved = TQ_METADATA_KIND_INT8;
			break;
		case DATEOID:
			resolved = TQ_METADATA_KIND_DATE;
			break;
		case TIMESTAMPTZOID:
			resolved = TQ_METADATA_KIND_TIMESTAMPTZ;
			break;
		case UUIDOID:
			resolved = TQ_METADATA_KIND_UUID;
			break;
		default:
			tq_metadata_error(errmsg, errmsg_len,
							  "turboquant metadata columns must be fixed-width bool, int2, int4, int8, date, timestamptz, or uuid");
			return false;
	}

	if (kind != NULL)
		*kind = resolved;
	return true;
}

bool
tq_metadata_attr_desc_init(Oid typid,
						   TqMetadataAttrDesc *desc,
						   char *errmsg,
						   size_t errmsg_len)
{
	if (desc == NULL)
		return false;
	if (!tq_metadata_kind_from_typid(typid, &desc->kind, errmsg, errmsg_len))
		return false;
	desc->typid = typid;
	return true;
}

size_t
tq_metadata_block_bytes(uint16_t attribute_count)
{
	return (size_t) TQ_METADATA_NULLMASK_BYTES
		+ ((size_t) attribute_count * (size_t) TQ_METADATA_SLOT_BYTES);
}

void
tq_metadata_zero_slots(uint8_t *slots, uint16_t attribute_count)
{
	if (slots == NULL)
		return;
	memset(slots, 0, (size_t) attribute_count * (size_t) TQ_METADATA_SLOT_BYTES);
}

bool
tq_metadata_encode_datum(TqMetadataKind kind,
						  Datum datum,
						  uint8_t slot[TQ_METADATA_SLOT_BYTES],
						  char *errmsg,
						  size_t errmsg_len)
{
	if (slot == NULL)
		return false;

	memset(slot, 0, TQ_METADATA_SLOT_BYTES);

	switch (kind)
	{
		case TQ_METADATA_KIND_BOOL:
			slot[0] = DatumGetBool(datum) ? 1u : 0u;
			return true;
		case TQ_METADATA_KIND_INT2:
		{
			int16 value = DatumGetInt16(datum);

			memcpy(slot, &value, sizeof(value));
			return true;
		}
		case TQ_METADATA_KIND_INT4:
		{
			int32 value = DatumGetInt32(datum);

			memcpy(slot, &value, sizeof(value));
			return true;
		}
		case TQ_METADATA_KIND_INT8:
		{
			int64 value = DatumGetInt64(datum);

			memcpy(slot, &value, sizeof(value));
			return true;
		}
		case TQ_METADATA_KIND_DATE:
		{
			DateADT value = DatumGetDateADT(datum);

			memcpy(slot, &value, sizeof(value));
			return true;
		}
		case TQ_METADATA_KIND_TIMESTAMPTZ:
		{
			TimestampTz value = DatumGetTimestampTz(datum);

			memcpy(slot, &value, sizeof(value));
			return true;
		}
		case TQ_METADATA_KIND_UUID:
		{
			const pg_uuid_t *uuid = DatumGetUUIDP(datum);

			memcpy(slot, uuid->data, UUID_LEN);
			return true;
		}
		case TQ_METADATA_KIND_INVALID:
		default:
			tq_metadata_error(errmsg, errmsg_len,
							  "invalid turboquant metadata kind");
			return false;
	}
}

bool
tq_metadata_decode_datum(TqMetadataKind kind,
						  const uint8_t slot[TQ_METADATA_SLOT_BYTES],
						  Datum *datum,
						  char *errmsg,
						  size_t errmsg_len)
{
	if (slot == NULL || datum == NULL)
		return false;

	switch (kind)
	{
		case TQ_METADATA_KIND_BOOL:
			*datum = BoolGetDatum(slot[0] != 0);
			return true;
		case TQ_METADATA_KIND_INT2:
		{
			int16 value = 0;

			memcpy(&value, slot, sizeof(value));
			*datum = Int16GetDatum(value);
			return true;
		}
		case TQ_METADATA_KIND_INT4:
		{
			int32 value = 0;

			memcpy(&value, slot, sizeof(value));
			*datum = Int32GetDatum(value);
			return true;
		}
		case TQ_METADATA_KIND_INT8:
		{
			int64 value = 0;

			memcpy(&value, slot, sizeof(value));
			*datum = Int64GetDatum(value);
			return true;
		}
		case TQ_METADATA_KIND_DATE:
		{
			DateADT value = 0;

			memcpy(&value, slot, sizeof(value));
			*datum = DateADTGetDatum(value);
			return true;
		}
		case TQ_METADATA_KIND_TIMESTAMPTZ:
		{
			TimestampTz value = 0;

			memcpy(&value, slot, sizeof(value));
			*datum = TimestampTzGetDatum(value);
			return true;
		}
		case TQ_METADATA_KIND_UUID:
		{
			pg_uuid_t *uuid = (pg_uuid_t *) palloc0(sizeof(pg_uuid_t));

			memcpy(uuid->data, slot, UUID_LEN);
			*datum = UUIDPGetDatum(uuid);
			return true;
		}
		case TQ_METADATA_KIND_INVALID:
		default:
			tq_metadata_error(errmsg, errmsg_len,
							  "invalid turboquant metadata kind");
			return false;
	}
}

bool
tq_metadata_slot_equals(TqMetadataKind kind,
						 const uint8_t left[TQ_METADATA_SLOT_BYTES],
						 const uint8_t right[TQ_METADATA_SLOT_BYTES],
						 char *errmsg,
						 size_t errmsg_len)
{
	(void) errmsg;
	(void) errmsg_len;

	if (left == NULL || right == NULL)
		return false;

	switch (kind)
	{
		case TQ_METADATA_KIND_BOOL:
			return left[0] == right[0];
		case TQ_METADATA_KIND_INT2:
			return memcmp(left, right, sizeof(int16)) == 0;
		case TQ_METADATA_KIND_INT4:
			return memcmp(left, right, sizeof(int32)) == 0;
		case TQ_METADATA_KIND_INT8:
		case TQ_METADATA_KIND_TIMESTAMPTZ:
			return memcmp(left, right, sizeof(int64)) == 0;
		case TQ_METADATA_KIND_DATE:
			return memcmp(left, right, sizeof(DateADT)) == 0;
		case TQ_METADATA_KIND_UUID:
			return memcmp(left, right, UUID_LEN) == 0;
		case TQ_METADATA_KIND_INVALID:
		default:
			return false;
	}
}

bool
tq_metadata_slot_compare(TqMetadataKind kind,
						  const uint8_t left[TQ_METADATA_SLOT_BYTES],
						  const uint8_t right[TQ_METADATA_SLOT_BYTES],
						  int *cmp,
						  char *errmsg,
						  size_t errmsg_len)
{
	if (left == NULL || right == NULL || cmp == NULL)
	{
		tq_metadata_error(errmsg, errmsg_len,
						  "invalid turboquant metadata comparison");
		return false;
	}

	*cmp = 0;

	switch (kind)
	{
		case TQ_METADATA_KIND_BOOL:
		{
			bool lhs = left[0] != 0;
			bool rhs = right[0] != 0;

			*cmp = (lhs > rhs) - (lhs < rhs);
			return true;
		}
		case TQ_METADATA_KIND_INT2:
		{
			int16 lhs = 0;
			int16 rhs = 0;

			memcpy(&lhs, left, sizeof(lhs));
			memcpy(&rhs, right, sizeof(rhs));
			*cmp = (lhs > rhs) - (lhs < rhs);
			return true;
		}
		case TQ_METADATA_KIND_INT4:
		{
			int32 lhs = 0;
			int32 rhs = 0;

			memcpy(&lhs, left, sizeof(lhs));
			memcpy(&rhs, right, sizeof(rhs));
			*cmp = (lhs > rhs) - (lhs < rhs);
			return true;
		}
		case TQ_METADATA_KIND_INT8:
		case TQ_METADATA_KIND_TIMESTAMPTZ:
		{
			int64 lhs = 0;
			int64 rhs = 0;

			memcpy(&lhs, left, sizeof(lhs));
			memcpy(&rhs, right, sizeof(rhs));
			*cmp = (lhs > rhs) - (lhs < rhs);
			return true;
		}
		case TQ_METADATA_KIND_DATE:
		{
			DateADT lhs = 0;
			DateADT rhs = 0;

			memcpy(&lhs, left, sizeof(lhs));
			memcpy(&rhs, right, sizeof(rhs));
			*cmp = (lhs > rhs) - (lhs < rhs);
			return true;
		}
		case TQ_METADATA_KIND_UUID:
			*cmp = memcmp(left, right, UUID_LEN);
			return true;
		case TQ_METADATA_KIND_INVALID:
		default:
			tq_metadata_error(errmsg, errmsg_len,
							  "invalid turboquant metadata kind");
			return false;
	}
}
