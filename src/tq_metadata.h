#ifndef TQ_METADATA_H
#define TQ_METADATA_H

#include "postgres.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/date.h"
#include "utils/timestamp.h"

#define TQ_MAX_STORED_METADATA_ATTRIBUTES 8
#define TQ_METADATA_SLOT_BYTES 16
#define TQ_METADATA_NULLMASK_BYTES 2

typedef enum TqMetadataKind
{
	TQ_METADATA_KIND_INVALID = 0,
	TQ_METADATA_KIND_BOOL,
	TQ_METADATA_KIND_INT2,
	TQ_METADATA_KIND_INT4,
	TQ_METADATA_KIND_INT8,
	TQ_METADATA_KIND_DATE,
	TQ_METADATA_KIND_TIMESTAMPTZ,
	TQ_METADATA_KIND_UUID
} TqMetadataKind;

typedef struct TqMetadataAttrDesc
{
	Oid			 typid;
	TqMetadataKind kind;
} TqMetadataAttrDesc;

extern bool tq_metadata_kind_from_typid(Oid typid,
										 TqMetadataKind *kind,
										 char *errmsg,
										 size_t errmsg_len);
extern bool tq_metadata_attr_desc_init(Oid typid,
									   TqMetadataAttrDesc *desc,
									   char *errmsg,
									   size_t errmsg_len);
extern size_t tq_metadata_block_bytes(uint16_t attribute_count);
extern void tq_metadata_zero_slots(uint8_t *slots, uint16_t attribute_count);
extern bool tq_metadata_encode_datum(TqMetadataKind kind,
									 Datum datum,
									 uint8_t slot[TQ_METADATA_SLOT_BYTES],
									 char *errmsg,
									 size_t errmsg_len);
extern bool tq_metadata_decode_datum(TqMetadataKind kind,
									 const uint8_t slot[TQ_METADATA_SLOT_BYTES],
									 Datum *datum,
									 char *errmsg,
									 size_t errmsg_len);
extern bool tq_metadata_slot_equals(TqMetadataKind kind,
									const uint8_t left[TQ_METADATA_SLOT_BYTES],
									const uint8_t right[TQ_METADATA_SLOT_BYTES],
									char *errmsg,
									size_t errmsg_len);

#endif
