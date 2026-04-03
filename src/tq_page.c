#include "src/tq_page.h"

#ifdef TQ_UNIT_TEST
#include <stdlib.h>
#else
#include "postgres.h"
#include "utils/palloc.h"
#endif

#include <stdio.h>
#include <string.h>

#define TQ_META_PAGE_HEADER_BYTES 122
#define TQ_LIST_DIR_PAGE_HEADER_BYTES 16
#define TQ_LIST_DIR_ENTRY_BYTES 32
#define TQ_BATCH_PAGE_HEADER_BYTES 44
#define TQ_CENTROID_PAGE_HEADER_BYTES 20
#define TQ_BATCH_SUMMARY_PAGE_HEADER_BYTES 20
#define TQ_TID_STORAGE_BYTES 6

#define TQ_FLAG_NORMALIZED UINT16_C(0x0001)
#define TQ_BATCH_FLAG_SOA_NIBBLES UINT16_C(0x0002)
#define TQ_BATCH_FLAG_LAYOUT_MASK UINT16_C(0x00FF)
#define TQ_BATCH_FLAG_INT4_ATTR_SHIFT 8
#define TQ_BATCH_FLAG_INT4_ATTR_MASK UINT16_C(0xFF00)

#define TQ_PAGE_MAGIC_OFFSET 0
#define TQ_PAGE_KIND_OFFSET 4
#define TQ_PAGE_HEADER_BYTES_OFFSET 6

#define TQ_META_VERSION_OFFSET 8
#define TQ_META_DIMENSION_OFFSET 12
#define TQ_META_TRANSFORM_OUTPUT_DIMENSION_OFFSET 16
#define TQ_META_CODEC_OFFSET 20
#define TQ_META_DISTANCE_OFFSET 22
#define TQ_META_BITS_OFFSET 24
#define TQ_META_LANE_COUNT_OFFSET 26
#define TQ_META_TRANSFORM_OFFSET 28
#define TQ_META_TRANSFORM_VERSION_OFFSET 30
#define TQ_META_FLAGS_OFFSET 32
#define TQ_META_LIST_COUNT_OFFSET 36
#define TQ_META_DIRECTORY_ROOT_OFFSET 40
#define TQ_META_CENTROID_ROOT_OFFSET 44
#define TQ_META_TRANSFORM_SEED_OFFSET 48
#define TQ_META_ROUTER_SEED_OFFSET 56
#define TQ_META_ROUTER_SAMPLE_COUNT_OFFSET 60
#define TQ_META_ROUTER_MAX_ITERATIONS_OFFSET 64
#define TQ_META_ROUTER_COMPLETED_ITERATIONS_OFFSET 68
#define TQ_META_ROUTER_TRAINED_VECTOR_COUNT_OFFSET 72
#define TQ_META_ROUTER_ALGORITHM_OFFSET 76
#define TQ_META_ROUTER_RESTART_COUNT_OFFSET 80
#define TQ_META_ROUTER_SELECTED_RESTART_OFFSET 84
#define TQ_META_ROUTER_MEAN_DISTORTION_OFFSET 88
#define TQ_META_ROUTER_MAX_LIST_OVER_AVG_OFFSET 92
#define TQ_META_ROUTER_COEFF_VAR_OFFSET 96
#define TQ_META_ROUTER_BALANCE_PENALTY_OFFSET 100
#define TQ_META_ROUTER_SELECTION_SCORE_OFFSET 104
#define TQ_META_ALGORITHM_VERSION_OFFSET 108
#define TQ_META_QUANTIZER_VERSION_OFFSET 110
#define TQ_META_RESIDUAL_SKETCH_VERSION_OFFSET 112
#define TQ_META_RESIDUAL_BITS_PER_DIMENSION_OFFSET 114
#define TQ_META_RESIDUAL_SKETCH_DIMENSION_OFFSET 116
#define TQ_META_ESTIMATOR_VERSION_OFFSET 120

#define TQ_LIST_DIR_ENTRY_CAPACITY_OFFSET 8
#define TQ_LIST_DIR_ENTRY_COUNT_OFFSET 10
#define TQ_LIST_DIR_NEXT_BLOCK_OFFSET 12

#define TQ_BATCH_LANE_COUNT_OFFSET 8
#define TQ_BATCH_OCCUPIED_COUNT_OFFSET 10
#define TQ_BATCH_LIVE_COUNT_OFFSET 12
#define TQ_BATCH_REPRESENTATIVE_LANE_OFFSET 14
#define TQ_BATCH_LIST_ID_OFFSET 16
#define TQ_BATCH_NEXT_BLOCK_OFFSET 20
#define TQ_BATCH_CODE_BYTES_OFFSET 24
#define TQ_BATCH_BITMAP_OFFSET_OFFSET 28
#define TQ_BATCH_TID_OFFSET_OFFSET 30
#define TQ_BATCH_CODE_OFFSET_OFFSET 32
#define TQ_BATCH_TOTAL_BYTES_OFFSET 34
#define TQ_BATCH_RESIDUAL_RADIUS_OFFSET 36
#define TQ_BATCH_FLAGS_OFFSET 40

#define TQ_CENTROID_DIMENSION_OFFSET 8
#define TQ_CENTROID_CAPACITY_OFFSET 12
#define TQ_CENTROID_COUNT_OFFSET 14
#define TQ_CENTROID_NEXT_BLOCK_OFFSET 16

#define TQ_BATCH_SUMMARY_CODE_BYTES_OFFSET 8
#define TQ_BATCH_SUMMARY_ENTRY_CAPACITY_OFFSET 12
#define TQ_BATCH_SUMMARY_ENTRY_COUNT_OFFSET 14
#define TQ_BATCH_SUMMARY_NEXT_BLOCK_OFFSET 16

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static void
tq_write_u16(uint8_t *dst, size_t offset, uint16_t value)
{
	dst[offset] = (uint8_t) (value & 0xFFu);
	dst[offset + 1] = (uint8_t) ((value >> 8) & 0xFFu);
}

static uint16_t
tq_read_u16(const uint8_t *src, size_t offset)
{
	return (uint16_t) src[offset]
		| (uint16_t) ((uint16_t) src[offset + 1] << 8);
}

static void
tq_write_u32(uint8_t *dst, size_t offset, uint32_t value)
{
	dst[offset] = (uint8_t) (value & 0xFFu);
	dst[offset + 1] = (uint8_t) ((value >> 8) & 0xFFu);
	dst[offset + 2] = (uint8_t) ((value >> 16) & 0xFFu);
	dst[offset + 3] = (uint8_t) ((value >> 24) & 0xFFu);
}

static uint32_t
tq_read_u32(const uint8_t *src, size_t offset)
{
	return (uint32_t) src[offset]
		| ((uint32_t) src[offset + 1] << 8)
		| ((uint32_t) src[offset + 2] << 16)
		| ((uint32_t) src[offset + 3] << 24);
}

static void
tq_write_u64(uint8_t *dst, size_t offset, uint64_t value)
{
	size_t		i;

	for (i = 0; i < sizeof(uint64_t); i++)
		dst[offset + i] = (uint8_t) ((value >> (i * 8)) & UINT64_C(0xFF));
}

static uint64_t
tq_read_u64(const uint8_t *src, size_t offset)
{
	uint64_t	value = 0;
	size_t		i;

	for (i = 0; i < sizeof(uint64_t); i++)
		value |= ((uint64_t) src[offset + i]) << (i * 8);

	return value;
}

static void
tq_write_float32(uint8_t *dst, size_t offset, float value)
{
	uint32_t bits = 0;

	memcpy(&bits, &value, sizeof(bits));
	tq_write_u32(dst, offset, bits);
}

static float
tq_read_float32(const uint8_t *src, size_t offset)
{
	uint32_t bits = tq_read_u32(src, offset);
	float value = 0.0f;

	memcpy(&value, &bits, sizeof(value));
	return value;
}

static uint16_t
tq_batch_flag_int4_attribute_count(uint16_t flags)
{
	return (uint16_t) ((flags & TQ_BATCH_FLAG_INT4_ATTR_MASK) >> TQ_BATCH_FLAG_INT4_ATTR_SHIFT);
}

static uint16_t
tq_batch_make_flags(bool soa_nibbles, uint16_t int4_attribute_count)
{
	uint16_t flags = soa_nibbles ? TQ_BATCH_FLAG_SOA_NIBBLES : 0;

	return (uint16_t) (flags
					   | ((int4_attribute_count << TQ_BATCH_FLAG_INT4_ATTR_SHIFT)
						  & TQ_BATCH_FLAG_INT4_ATTR_MASK));
}

static bool
tq_validate_page_common(const uint8_t *page,
						size_t page_size,
						TqPageKind expected_kind,
						uint16_t expected_header_bytes,
						char *errmsg,
						size_t errmsg_len)
{
	if (page_size < expected_header_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant page: page buffer is too small");
		return false;
	}

	if (tq_read_u32(page, TQ_PAGE_MAGIC_OFFSET) != TQ_PAGE_MAGIC)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant page: bad magic");
		return false;
	}

	if (tq_read_u16(page, TQ_PAGE_KIND_OFFSET) != (uint16_t) expected_kind)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant page: unexpected page kind");
		return false;
	}

	if (tq_read_u16(page, TQ_PAGE_HEADER_BYTES_OFFSET) != expected_header_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant page: unexpected header size");
		return false;
	}

	return true;
}

static void
tq_write_page_common(uint8_t *page, TqPageKind kind, uint16_t header_bytes)
{
	tq_write_u32(page, TQ_PAGE_MAGIC_OFFSET, TQ_PAGE_MAGIC);
	tq_write_u16(page, TQ_PAGE_KIND_OFFSET, (uint16_t) kind);
	tq_write_u16(page, TQ_PAGE_HEADER_BYTES_OFFSET, header_bytes);
}

bool
tq_page_read_kind(const void *page,
				  size_t page_size,
				  TqPageKind *kind,
				  char *errmsg,
				  size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || kind == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant page: page and kind output must be non-null");
		return false;
	}

	if (page_size < TQ_BATCH_PAGE_HEADER_BYTES)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant page: page buffer is too small");
		return false;
	}

	if (tq_read_u32(bytes, TQ_PAGE_MAGIC_OFFSET) != TQ_PAGE_MAGIC)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant page: bad magic");
		return false;
	}

	*kind = (TqPageKind) tq_read_u16(bytes, TQ_PAGE_KIND_OFFSET);
	return true;
}

static size_t
tq_list_dir_entry_offset(uint16_t index)
{
	return TQ_LIST_DIR_PAGE_HEADER_BYTES
		+ ((size_t) index * (size_t) TQ_LIST_DIR_ENTRY_BYTES);
}

static bool
tq_list_dir_validate_index(const uint8_t *page,
						   size_t page_size,
						   uint16_t index,
						   char *errmsg,
						   size_t errmsg_len)
{
	uint16_t	entry_capacity = 0;
	size_t		entry_end = 0;

	if (!tq_validate_page_common(page, page_size, TQ_PAGE_KIND_LIST_DIRECTORY,
								 TQ_LIST_DIR_PAGE_HEADER_BYTES, errmsg, errmsg_len))
		return false;

	entry_capacity = tq_read_u16(page, TQ_LIST_DIR_ENTRY_CAPACITY_OFFSET);

	if (index >= entry_capacity)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant list directory page: entry index out of range");
		return false;
	}

	entry_end = tq_list_dir_entry_offset(index + 1);
	if (entry_end > page_size)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant list directory page: entry storage exceeds page size");
		return false;
	}

	return true;
}

/* Forward declarations for SoA offset helpers (defined after tq_bitmap_bytes_for_lanes) */
static size_t tq_batch_soa_tid_array_offset(uint16_t lane_count);
static size_t tq_batch_soa_filter_array_offset(uint16_t lane_count);
static size_t tq_batch_soa_gamma_array_offset(uint16_t lane_count, uint16_t int4_attribute_count);
static size_t tq_batch_soa_nibble_block_offset(uint16_t lane_count, uint16_t int4_attribute_count);
static size_t tq_batch_soa_representative_offset(uint16_t lane_count, uint32_t dimension,
												 uint16_t int4_attribute_count);
static bool tq_batch_is_soa_page(const uint8_t *page);
static size_t tq_batch_entry_stride_from_layout(const uint8_t *page);
static size_t tq_batch_required_bytes_internal(uint16_t lane_count,
											   uint32_t code_bytes,
											   uint16_t int4_attribute_count);
static size_t tq_batch_soa_required_bytes_internal(uint16_t lane_count,
												   uint32_t dimension,
												   uint32_t representative_code_bytes,
												   uint16_t int4_attribute_count);

static bool
tq_batch_validate_header(const uint8_t *page,
						 size_t page_size,
						 char *errmsg,
						 size_t errmsg_len)
{
	uint16_t	lane_count = 0;
	uint32_t	code_bytes = 0;
	uint16_t	total_bytes = 0;
	uint16_t	flags = 0;
	uint16_t	int4_attribute_count = 0;

	if (!tq_validate_page_common(page, page_size, TQ_PAGE_KIND_BATCH,
								 TQ_BATCH_PAGE_HEADER_BYTES, errmsg, errmsg_len))
		return false;

	lane_count = tq_read_u16(page, TQ_BATCH_LANE_COUNT_OFFSET);
	code_bytes = tq_read_u32(page, TQ_BATCH_CODE_BYTES_OFFSET);
	total_bytes = tq_read_u16(page, TQ_BATCH_TOTAL_BYTES_OFFSET);
	flags = tq_read_u16(page, TQ_BATCH_FLAGS_OFFSET);
	int4_attribute_count = tq_batch_flag_int4_attribute_count(flags);

	if (lane_count == 0 || code_bytes == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane count and code bytes must be positive");
		return false;
	}

	if ((flags & TQ_BATCH_FLAG_LAYOUT_MASK) != 0
		&& (flags & TQ_BATCH_FLAG_LAYOUT_MASK) != TQ_BATCH_FLAG_SOA_NIBBLES)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: unsupported flags");
		return false;
	}

	if (int4_attribute_count > TQ_MAX_STORED_INT4_ATTRIBUTES)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: unsupported int4 attribute count");
		return false;
	}

	if (flags & TQ_BATCH_FLAG_SOA_NIBBLES)
	{
		/*
		 * SoA layout: tid_offset stores the dimension (repurposed), code_offset
		 * is unused (set to 0).  total_bytes reflects the SoA footprint
		 * computed from lane_count, dimension, and code_bytes.
		 */
		uint16_t	dimension = tq_read_u16(page, TQ_BATCH_TID_OFFSET_OFFSET);
		size_t		expected_total;

		if (dimension == 0)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant batch page: SoA page must have positive dimension");
			return false;
		}

		expected_total = tq_batch_soa_representative_offset(lane_count, dimension,
															int4_attribute_count)
						 + (size_t) code_bytes;
		if ((size_t) total_bytes != expected_total)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant batch page: SoA total bytes do not match layout");
			return false;
		}
	}
	else
	{
		uint16_t	expected_tid_offset;
		uint16_t	expected_code_offset;

		if ((size_t) total_bytes != tq_batch_required_bytes_internal(
				lane_count,
				code_bytes,
				int4_attribute_count))
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant batch page: total bytes do not match layout");
			return false;
		}

		expected_tid_offset = (uint16_t) (TQ_BATCH_PAGE_HEADER_BYTES
										   + tq_bitmap_bytes_for_lanes(lane_count));
		expected_code_offset = (uint16_t) (expected_tid_offset
										   + TQ_TID_STORAGE_BYTES
										   + ((size_t) int4_attribute_count * sizeof(int32_t)));

		if (tq_read_u16(page, TQ_BATCH_TID_OFFSET_OFFSET) != expected_tid_offset)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant batch page: unexpected entry offset");
			return false;
		}

		if (tq_read_u16(page, TQ_BATCH_CODE_OFFSET_OFFSET) != expected_code_offset)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant batch page: unexpected code offset");
			return false;
		}
	}

	if ((size_t) total_bytes > page_size)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: layout exceeds page size");
		return false;
	}

	if (tq_read_u16(page, TQ_BATCH_OCCUPIED_COUNT_OFFSET) > lane_count)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: occupied lanes exceed lane count");
		return false;
	}

	if (tq_read_u16(page, TQ_BATCH_LIVE_COUNT_OFFSET) > tq_read_u16(page, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: live lanes exceed occupied lanes");
		return false;
	}

	if (tq_read_u16(page, TQ_BATCH_REPRESENTATIVE_LANE_OFFSET) != TQ_BATCH_PAGE_NO_REPRESENTATIVE
		&& tq_read_u16(page, TQ_BATCH_REPRESENTATIVE_LANE_OFFSET) >= tq_read_u16(page, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: representative lane exceeds occupied lanes");
		return false;
	}

	if (tq_read_u16(page, TQ_BATCH_BITMAP_OFFSET_OFFSET) != TQ_BATCH_PAGE_HEADER_BYTES)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: unexpected bitmap offset");
		return false;
	}

	return true;
}

static size_t
tq_centroid_offset(const uint8_t *page, uint16_t index)
{
	uint32_t	dimension = tq_read_u32(page, TQ_CENTROID_DIMENSION_OFFSET);

	return (size_t) TQ_CENTROID_PAGE_HEADER_BYTES
		+ ((size_t) index * (size_t) dimension * sizeof(float));
}

static bool
tq_centroid_validate_header(const uint8_t *page,
							size_t page_size,
							char *errmsg,
							size_t errmsg_len)
{
	uint32_t	dimension = 0;
	uint16_t	centroid_capacity = 0;
	size_t		required_bytes = 0;

	if (!tq_validate_page_common(page, page_size, TQ_PAGE_KIND_CENTROID,
								 TQ_CENTROID_PAGE_HEADER_BYTES, errmsg, errmsg_len))
		return false;

	dimension = tq_read_u32(page, TQ_CENTROID_DIMENSION_OFFSET);
	centroid_capacity = tq_read_u16(page, TQ_CENTROID_CAPACITY_OFFSET);

	if (dimension == 0 || centroid_capacity == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: dimension and capacity must be positive");
		return false;
	}

	required_bytes = tq_centroid_page_required_bytes(dimension, centroid_capacity);
	if (required_bytes > page_size)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: layout exceeds page size");
		return false;
	}

	if (tq_read_u16(page, TQ_CENTROID_COUNT_OFFSET) > centroid_capacity)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: centroid count exceeds capacity");
		return false;
	}

	return true;
}

static size_t
tq_batch_summary_entry_bytes(uint32_t code_bytes)
{
	return (size_t) 12 + (size_t) code_bytes;
}

static size_t
tq_batch_summary_entry_offset(const uint8_t *page, uint16_t index)
{
	return (size_t) TQ_BATCH_SUMMARY_PAGE_HEADER_BYTES
		+ ((size_t) index * tq_batch_summary_entry_bytes(tq_read_u32(page, TQ_BATCH_SUMMARY_CODE_BYTES_OFFSET)));
}

static bool
tq_batch_summary_validate_header(const uint8_t *page,
								 size_t page_size,
								 char *errmsg,
								 size_t errmsg_len)
{
	uint32_t	code_bytes = 0;
	uint16_t	entry_capacity = 0;
	size_t		required_bytes = 0;

	if (!tq_validate_page_common(page, page_size, TQ_PAGE_KIND_BATCH_SUMMARY,
								 TQ_BATCH_SUMMARY_PAGE_HEADER_BYTES, errmsg, errmsg_len))
		return false;

	code_bytes = tq_read_u32(page, TQ_BATCH_SUMMARY_CODE_BYTES_OFFSET);
	entry_capacity = tq_read_u16(page, TQ_BATCH_SUMMARY_ENTRY_CAPACITY_OFFSET);
	if (code_bytes == 0 || entry_capacity == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: code bytes and capacity must be positive");
		return false;
	}

	required_bytes = tq_batch_summary_page_required_bytes(entry_capacity, code_bytes);
	if (required_bytes > page_size)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: layout exceeds page size");
		return false;
	}

	if (tq_read_u16(page, TQ_BATCH_SUMMARY_ENTRY_COUNT_OFFSET) > entry_capacity)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: entry count exceeds capacity");
		return false;
	}

	return true;
}

static uint8_t *
tq_batch_bitmap(uint8_t *page)
{
	return page + tq_read_u16(page, TQ_BATCH_BITMAP_OFFSET_OFFSET);
}

static const uint8_t *
tq_batch_bitmap_const(const uint8_t *page)
{
	return page + tq_read_u16(page, TQ_BATCH_BITMAP_OFFSET_OFFSET);
}

static size_t
tq_batch_tid_offset(const uint8_t *page, uint16_t lane_index)
{
	size_t		entry_offset = (size_t) tq_read_u16(page, TQ_BATCH_TID_OFFSET_OFFSET);
	size_t		entry_stride = tq_batch_entry_stride_from_layout(page);

	return entry_offset + (entry_stride * (size_t) lane_index);
}

static size_t
tq_batch_code_offset(const uint8_t *page, uint16_t lane_index)
{
	size_t tid_offset = (size_t) tq_read_u16(page, TQ_BATCH_TID_OFFSET_OFFSET);
	size_t code_offset = (size_t) tq_read_u16(page, TQ_BATCH_CODE_OFFSET_OFFSET);

	return tq_batch_tid_offset(page, lane_index) + (code_offset - tid_offset);
}

static size_t
tq_batch_filter_int4_offset(const uint8_t *page, uint16_t lane_index)
{
	return tq_batch_tid_offset(page, lane_index) + (size_t) TQ_TID_STORAGE_BYTES;
}

static size_t
tq_batch_int4_attribute_offset(const uint8_t *page,
								   uint16_t lane_index,
								   uint16_t attribute_index)
{
	return tq_batch_filter_int4_offset(page, lane_index)
		+ ((size_t) attribute_index * sizeof(int32_t));
}

static size_t
tq_batch_soa_tid_array_offset(uint16_t lane_count)
{
	return TQ_BATCH_PAGE_HEADER_BYTES + tq_bitmap_bytes_for_lanes(lane_count);
}

static size_t
tq_batch_soa_filter_array_offset(uint16_t lane_count)
{
	return tq_batch_soa_tid_array_offset(lane_count)
		+ (size_t) lane_count * TQ_TID_STORAGE_BYTES;
}

static size_t
tq_batch_soa_int4_attribute_offset(uint16_t lane_count,
								   uint16_t lane_index,
								   uint16_t attribute_index)
{
	return tq_batch_soa_filter_array_offset(lane_count)
		+ ((((size_t) attribute_index * (size_t) lane_count) + (size_t) lane_index)
		   * sizeof(int32_t));
}

static size_t
tq_batch_soa_gamma_array_offset(uint16_t lane_count, uint16_t int4_attribute_count)
{
	return tq_batch_soa_filter_array_offset(lane_count)
		+ ((size_t) lane_count * (size_t) int4_attribute_count * sizeof(int32_t));
}

static size_t
tq_batch_soa_nibble_block_offset(uint16_t lane_count, uint16_t int4_attribute_count)
{
	return tq_batch_soa_gamma_array_offset(lane_count, int4_attribute_count)
		+ (size_t) lane_count * sizeof(float);
}

static size_t
tq_batch_soa_representative_offset(uint16_t lane_count, uint32_t dimension,
								   uint16_t int4_attribute_count)
{
	size_t	nibble_pair_cols = ((size_t) lane_count + 1u) / 2u;

	return tq_batch_soa_nibble_block_offset(lane_count, int4_attribute_count)
		+ (size_t) dimension * nibble_pair_cols;
}

static bool
tq_batch_is_soa_page(const uint8_t *page)
{
	return (tq_read_u16(page, TQ_BATCH_FLAGS_OFFSET) & TQ_BATCH_FLAG_SOA_NIBBLES) != 0;
}

static size_t
tq_batch_entry_stride_from_layout(const uint8_t *page)
{
	size_t tid_offset = (size_t) tq_read_u16(page, TQ_BATCH_TID_OFFSET_OFFSET);
	size_t code_offset = (size_t) tq_read_u16(page, TQ_BATCH_CODE_OFFSET_OFFSET);

	return (code_offset - tid_offset) + (size_t) tq_read_u32(page, TQ_BATCH_CODE_BYTES_OFFSET);
}

static bool
tq_batch_lane_is_live(const uint8_t *page, uint16_t lane_index)
{
	const uint8_t *bitmap = tq_batch_bitmap_const(page);
	uint8_t		mask = (uint8_t) (1u << (lane_index % 8u));

	return (bitmap[lane_index / 8u] & mask) != 0;
}

static void
tq_batch_set_live(uint8_t *page, uint16_t lane_index, bool is_live)
{
	uint8_t    *bitmap = tq_batch_bitmap(page);
	uint8_t		mask = (uint8_t) (1u << (lane_index % 8u));

	if (is_live)
		bitmap[lane_index / 8u] |= mask;
	else
		bitmap[lane_index / 8u] &= (uint8_t) ~mask;
}

static void
tq_write_tid(uint8_t *page, uint16_t lane_index, const TqTid *tid)
{
	size_t		offset = tq_batch_tid_offset(page, lane_index);

	tq_write_u32(page, offset, tid->block_number);
	tq_write_u16(page, offset + 4, tid->offset_number);
}

static void
tq_read_tid(const uint8_t *page, uint16_t lane_index, TqTid *tid)
{
	size_t		offset = tq_batch_tid_offset(page, lane_index);

	tid->block_number = tq_read_u32(page, offset);
	tid->offset_number = tq_read_u16(page, offset + 4);
}

size_t
tq_bitmap_bytes_for_lanes(uint16_t lane_count)
{
	return ((size_t) lane_count + 7u) / 8u;
}

size_t
tq_batch_page_required_bytes(uint16_t lane_count, uint32_t code_bytes)
{
	return tq_batch_required_bytes_internal(lane_count, code_bytes, false);
}

static size_t
tq_batch_required_bytes_internal(uint16_t lane_count,
								 uint32_t code_bytes,
								 uint16_t int4_attribute_count)
{
	size_t per_lane_bytes = (size_t) TQ_TID_STORAGE_BYTES
		+ ((size_t) int4_attribute_count * sizeof(int32_t))
		+ (size_t) code_bytes;

	return (size_t) TQ_BATCH_PAGE_HEADER_BYTES
		+ tq_bitmap_bytes_for_lanes(lane_count)
		+ ((size_t) lane_count * per_lane_bytes);
}

static size_t
tq_batch_soa_required_bytes_internal(uint16_t lane_count,
									 uint32_t dimension,
									 uint32_t representative_code_bytes,
									 uint16_t int4_attribute_count)
{
	return tq_batch_soa_representative_offset(lane_count, dimension, int4_attribute_count)
		+ (size_t) representative_code_bytes;
}

bool
tq_batch_page_can_fit(size_t page_size, uint16_t lane_count, uint32_t code_bytes)
{
	if (lane_count == 0 || code_bytes == 0)
		return false;

	return tq_batch_page_required_bytes(lane_count, code_bytes) <= page_size;
}

uint16_t
tq_list_dir_page_capacity(size_t page_size)
{
	if (page_size <= TQ_LIST_DIR_PAGE_HEADER_BYTES)
		return 0;

	return (uint16_t) ((page_size - TQ_LIST_DIR_PAGE_HEADER_BYTES)
					   / (size_t) TQ_LIST_DIR_ENTRY_BYTES);
}

size_t
tq_centroid_page_required_bytes(uint32_t dimension, uint16_t centroid_capacity)
{
	return (size_t) TQ_CENTROID_PAGE_HEADER_BYTES
		+ ((size_t) centroid_capacity * (size_t) dimension * sizeof(float));
}

uint16_t
tq_centroid_page_capacity(size_t page_size, uint32_t dimension)
{
	if (dimension == 0 || page_size <= TQ_CENTROID_PAGE_HEADER_BYTES)
		return 0;

	return (uint16_t) ((page_size - TQ_CENTROID_PAGE_HEADER_BYTES)
					   / ((size_t) dimension * sizeof(float)));
}

size_t
tq_batch_summary_page_required_bytes(uint16_t entry_capacity, uint32_t code_bytes)
{
	return (size_t) TQ_BATCH_SUMMARY_PAGE_HEADER_BYTES
		+ ((size_t) entry_capacity * tq_batch_summary_entry_bytes(code_bytes));
}

uint16_t
tq_batch_summary_page_capacity(size_t page_size, uint32_t code_bytes)
{
	if (code_bytes == 0 || page_size <= TQ_BATCH_SUMMARY_PAGE_HEADER_BYTES)
		return 0;

	return (uint16_t) ((page_size - TQ_BATCH_SUMMARY_PAGE_HEADER_BYTES)
					   / tq_batch_summary_entry_bytes(code_bytes));
}

bool
tq_meta_page_init(void *page,
				  size_t page_size,
				  const TqMetaPageFields *fields,
				  char *errmsg,
				  size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	uint16_t	flags = 0;

	if (page == NULL || fields == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant meta page: page and fields must be non-null");
		return false;
	}

	if (page_size < TQ_META_PAGE_HEADER_BYTES)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant meta page: page buffer is too small");
		return false;
	}

	memset(page, 0, page_size);
	tq_write_page_common(bytes, TQ_PAGE_KIND_META, TQ_META_PAGE_HEADER_BYTES);
	tq_write_u32(bytes, TQ_META_VERSION_OFFSET, TQ_PAGE_FORMAT_VERSION);
	tq_write_u32(bytes, TQ_META_DIMENSION_OFFSET, fields->dimension);
	tq_write_u32(bytes, TQ_META_TRANSFORM_OUTPUT_DIMENSION_OFFSET, fields->transform_output_dimension);
	tq_write_u16(bytes, TQ_META_CODEC_OFFSET, (uint16_t) fields->codec);
	tq_write_u16(bytes, TQ_META_DISTANCE_OFFSET, (uint16_t) fields->distance);
	tq_write_u16(bytes, TQ_META_BITS_OFFSET, fields->bits);
	tq_write_u16(bytes, TQ_META_LANE_COUNT_OFFSET, fields->lane_count);
	tq_write_u16(bytes, TQ_META_TRANSFORM_OFFSET, (uint16_t) fields->transform);
	tq_write_u16(bytes, TQ_META_TRANSFORM_VERSION_OFFSET, fields->transform_version);

	if (fields->normalized)
		flags |= TQ_FLAG_NORMALIZED;

	tq_write_u16(bytes, TQ_META_FLAGS_OFFSET, flags);
	tq_write_u32(bytes, TQ_META_LIST_COUNT_OFFSET, fields->list_count);
	tq_write_u32(bytes, TQ_META_DIRECTORY_ROOT_OFFSET, fields->directory_root_block);
	tq_write_u32(bytes, TQ_META_CENTROID_ROOT_OFFSET, fields->centroid_root_block);
	tq_write_u64(bytes, TQ_META_TRANSFORM_SEED_OFFSET, fields->transform_seed);
	tq_write_u32(bytes, TQ_META_ROUTER_SEED_OFFSET, fields->router_seed);
	tq_write_u32(bytes, TQ_META_ROUTER_SAMPLE_COUNT_OFFSET, fields->router_sample_count);
	tq_write_u32(bytes, TQ_META_ROUTER_MAX_ITERATIONS_OFFSET, fields->router_max_iterations);
	tq_write_u32(bytes, TQ_META_ROUTER_COMPLETED_ITERATIONS_OFFSET, fields->router_completed_iterations);
	tq_write_u32(bytes, TQ_META_ROUTER_TRAINED_VECTOR_COUNT_OFFSET, fields->router_trained_vector_count);
	tq_write_u16(bytes, TQ_META_ROUTER_ALGORITHM_OFFSET, (uint16_t) fields->router_algorithm);
	tq_write_u32(bytes, TQ_META_ROUTER_RESTART_COUNT_OFFSET, fields->router_restart_count);
	tq_write_u32(bytes, TQ_META_ROUTER_SELECTED_RESTART_OFFSET, fields->router_selected_restart);
	tq_write_float32(bytes, TQ_META_ROUTER_MEAN_DISTORTION_OFFSET, fields->router_mean_distortion);
	tq_write_float32(bytes, TQ_META_ROUTER_MAX_LIST_OVER_AVG_OFFSET, fields->router_max_list_over_avg);
	tq_write_float32(bytes, TQ_META_ROUTER_COEFF_VAR_OFFSET, fields->router_coeff_var);
	tq_write_float32(bytes, TQ_META_ROUTER_BALANCE_PENALTY_OFFSET, fields->router_balance_penalty);
	tq_write_float32(bytes, TQ_META_ROUTER_SELECTION_SCORE_OFFSET, fields->router_selection_score);
	tq_write_u16(bytes, TQ_META_ALGORITHM_VERSION_OFFSET, fields->algorithm_version);
	tq_write_u16(bytes, TQ_META_QUANTIZER_VERSION_OFFSET, fields->quantizer_version);
	tq_write_u16(bytes, TQ_META_RESIDUAL_SKETCH_VERSION_OFFSET, fields->residual_sketch_version);
	tq_write_u16(bytes, TQ_META_RESIDUAL_BITS_PER_DIMENSION_OFFSET, fields->residual_bits_per_dimension);
	tq_write_u32(bytes, TQ_META_RESIDUAL_SKETCH_DIMENSION_OFFSET, fields->residual_sketch_dimension);
	tq_write_u16(bytes, TQ_META_ESTIMATOR_VERSION_OFFSET, fields->estimator_version);
	return true;
}

bool
tq_meta_page_read(const void *page,
				  size_t page_size,
				  TqMetaPageFields *fields,
				  char *errmsg,
				  size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || fields == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant meta page: page and fields must be non-null");
		return false;
	}

	if (!tq_validate_page_common(bytes, page_size, TQ_PAGE_KIND_META,
								 TQ_META_PAGE_HEADER_BYTES, errmsg, errmsg_len))
		return false;

	if (tq_read_u32(bytes, TQ_META_VERSION_OFFSET) != TQ_PAGE_FORMAT_VERSION)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant meta page: unsupported format version");
		return false;
	}

	fields->dimension = tq_read_u32(bytes, TQ_META_DIMENSION_OFFSET);
	fields->transform_output_dimension = tq_read_u32(bytes, TQ_META_TRANSFORM_OUTPUT_DIMENSION_OFFSET);
	fields->codec = (TqCodecKind) tq_read_u16(bytes, TQ_META_CODEC_OFFSET);
	fields->distance = (TqDistanceKind) tq_read_u16(bytes, TQ_META_DISTANCE_OFFSET);
	fields->bits = tq_read_u16(bytes, TQ_META_BITS_OFFSET);
	fields->lane_count = tq_read_u16(bytes, TQ_META_LANE_COUNT_OFFSET);
	fields->transform = (TqTransformKind) tq_read_u16(bytes, TQ_META_TRANSFORM_OFFSET);
	fields->transform_version = tq_read_u16(bytes, TQ_META_TRANSFORM_VERSION_OFFSET);
	fields->normalized = (tq_read_u16(bytes, TQ_META_FLAGS_OFFSET) & TQ_FLAG_NORMALIZED) != 0;
	if ((tq_read_u16(bytes, TQ_META_FLAGS_OFFSET) & ~TQ_FLAG_NORMALIZED) != 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant meta page: unsupported flags");
		return false;
	}
	fields->list_count = tq_read_u32(bytes, TQ_META_LIST_COUNT_OFFSET);
	fields->directory_root_block = tq_read_u32(bytes, TQ_META_DIRECTORY_ROOT_OFFSET);
	fields->centroid_root_block = tq_read_u32(bytes, TQ_META_CENTROID_ROOT_OFFSET);
	fields->transform_seed = tq_read_u64(bytes, TQ_META_TRANSFORM_SEED_OFFSET);
	fields->router_seed = tq_read_u32(bytes, TQ_META_ROUTER_SEED_OFFSET);
	fields->router_sample_count = tq_read_u32(bytes, TQ_META_ROUTER_SAMPLE_COUNT_OFFSET);
	fields->router_max_iterations = tq_read_u32(bytes, TQ_META_ROUTER_MAX_ITERATIONS_OFFSET);
	fields->router_completed_iterations = tq_read_u32(bytes, TQ_META_ROUTER_COMPLETED_ITERATIONS_OFFSET);
	fields->router_trained_vector_count = tq_read_u32(bytes, TQ_META_ROUTER_TRAINED_VECTOR_COUNT_OFFSET);
	fields->router_algorithm = (TqRouterAlgorithmKind) tq_read_u16(bytes, TQ_META_ROUTER_ALGORITHM_OFFSET);
	fields->router_restart_count = tq_read_u32(bytes, TQ_META_ROUTER_RESTART_COUNT_OFFSET);
	fields->router_selected_restart = tq_read_u32(bytes, TQ_META_ROUTER_SELECTED_RESTART_OFFSET);
	fields->router_mean_distortion = tq_read_float32(bytes, TQ_META_ROUTER_MEAN_DISTORTION_OFFSET);
	fields->router_max_list_over_avg = tq_read_float32(bytes, TQ_META_ROUTER_MAX_LIST_OVER_AVG_OFFSET);
	fields->router_coeff_var = tq_read_float32(bytes, TQ_META_ROUTER_COEFF_VAR_OFFSET);
	fields->router_balance_penalty = tq_read_float32(bytes, TQ_META_ROUTER_BALANCE_PENALTY_OFFSET);
	fields->router_selection_score = tq_read_float32(bytes, TQ_META_ROUTER_SELECTION_SCORE_OFFSET);
	fields->algorithm_version = tq_read_u16(bytes, TQ_META_ALGORITHM_VERSION_OFFSET);
	fields->quantizer_version = tq_read_u16(bytes, TQ_META_QUANTIZER_VERSION_OFFSET);
	fields->residual_sketch_version = tq_read_u16(bytes, TQ_META_RESIDUAL_SKETCH_VERSION_OFFSET);
	fields->residual_bits_per_dimension = tq_read_u16(bytes, TQ_META_RESIDUAL_BITS_PER_DIMENSION_OFFSET);
	fields->residual_sketch_dimension = tq_read_u32(bytes, TQ_META_RESIDUAL_SKETCH_DIMENSION_OFFSET);
	fields->estimator_version = tq_read_u16(bytes, TQ_META_ESTIMATOR_VERSION_OFFSET);
	if (fields->transform_version != TQ_TRANSFORM_CONTRACT_VERSION)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant meta page: unsupported transform contract");
		return false;
	}
	if (fields->dimension == 0)
	{
		if (fields->transform_output_dimension != 0)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant meta page: empty indexes must not store transform output dimension");
			return false;
		}
		return true;
	}
	if (fields->transform_output_dimension != tq_transform_padded_dimension(fields->dimension))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant meta page: transform output dimension does not match persisted contract");
		return false;
	}
	if (fields->algorithm_version != TQ_ALGORITHM_VERSION
		|| fields->quantizer_version != TQ_QUANTIZER_VERSION
		|| fields->residual_sketch_version != TQ_RESIDUAL_SKETCH_VERSION
		|| fields->residual_bits_per_dimension != 1
		|| fields->residual_sketch_dimension == 0
		|| fields->residual_sketch_dimension > fields->transform_output_dimension
		|| fields->estimator_version != TQ_ESTIMATOR_VERSION)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant meta page: unsupported faithful turboquant metadata");
		return false;
	}
	return true;
}

bool
tq_list_dir_page_init(void *page,
					  size_t page_size,
					  uint16_t entry_capacity,
					  uint32_t next_block,
					  char *errmsg,
					  size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	size_t		required_bytes = 0;

	if (page == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant list directory page: page must be non-null");
		return false;
	}

	required_bytes = TQ_LIST_DIR_PAGE_HEADER_BYTES
		+ ((size_t) entry_capacity * (size_t) TQ_LIST_DIR_ENTRY_BYTES);

	if (page_size < required_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant list directory page: page buffer is too small for requested capacity");
		return false;
	}

	memset(page, 0, page_size);
	tq_write_page_common(bytes, TQ_PAGE_KIND_LIST_DIRECTORY,
						 TQ_LIST_DIR_PAGE_HEADER_BYTES);
	tq_write_u16(bytes, TQ_LIST_DIR_ENTRY_CAPACITY_OFFSET, entry_capacity);
	tq_write_u16(bytes, TQ_LIST_DIR_ENTRY_COUNT_OFFSET, 0);
	tq_write_u32(bytes, TQ_LIST_DIR_NEXT_BLOCK_OFFSET, next_block);
	return true;
}

bool
tq_list_dir_page_read_header(const void *page,
							 size_t page_size,
							 TqListDirPageHeaderView *header,
							 char *errmsg,
							 size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || header == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant list directory page: page and header must be non-null");
		return false;
	}

	if (!tq_validate_page_common(bytes, page_size, TQ_PAGE_KIND_LIST_DIRECTORY,
								 TQ_LIST_DIR_PAGE_HEADER_BYTES, errmsg, errmsg_len))
		return false;

	header->entry_capacity = tq_read_u16(bytes, TQ_LIST_DIR_ENTRY_CAPACITY_OFFSET);
	header->entry_count = tq_read_u16(bytes, TQ_LIST_DIR_ENTRY_COUNT_OFFSET);
	header->next_block = tq_read_u32(bytes, TQ_LIST_DIR_NEXT_BLOCK_OFFSET);
	return true;
}

bool
tq_list_dir_page_set_entry(void *page,
						   size_t page_size,
						   uint16_t index,
						   const TqListDirEntry *entry,
						   char *errmsg,
						   size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	size_t		offset = 0;
	uint16_t	entry_count = 0;

	if (page == NULL || entry == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant list directory page: page and entry must be non-null");
		return false;
	}

	if (!tq_list_dir_validate_index(bytes, page_size, index, errmsg, errmsg_len))
		return false;

	offset = tq_list_dir_entry_offset(index);
	tq_write_u32(bytes, offset + 0, entry->list_id);
	tq_write_u32(bytes, offset + 4, entry->head_block);
	tq_write_u32(bytes, offset + 8, entry->tail_block);
	tq_write_u32(bytes, offset + 12, entry->live_count);
	tq_write_u32(bytes, offset + 16, entry->dead_count);
	tq_write_u32(bytes, offset + 20, entry->batch_page_count);
	tq_write_u32(bytes, offset + 24, entry->summary_head_block);
	tq_write_u16(bytes, offset + 28, entry->free_lane_hint);
	tq_write_u16(bytes, offset + 30, 0);

	entry_count = tq_read_u16(bytes, TQ_LIST_DIR_ENTRY_COUNT_OFFSET);
	if (index >= entry_count)
		tq_write_u16(bytes, TQ_LIST_DIR_ENTRY_COUNT_OFFSET, (uint16_t) (index + 1));

	return true;
}

bool
tq_list_dir_page_get_entry(const void *page,
						   size_t page_size,
						   uint16_t index,
						   TqListDirEntry *entry,
						   char *errmsg,
						   size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	size_t		offset = 0;

	if (page == NULL || entry == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant list directory page: page and entry must be non-null");
		return false;
	}

	if (!tq_list_dir_validate_index(bytes, page_size, index, errmsg, errmsg_len))
		return false;

	if (index >= tq_read_u16(bytes, TQ_LIST_DIR_ENTRY_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant list directory page: entry has not been initialized");
		return false;
	}

	offset = tq_list_dir_entry_offset(index);
	entry->list_id = tq_read_u32(bytes, offset + 0);
	entry->head_block = tq_read_u32(bytes, offset + 4);
	entry->tail_block = tq_read_u32(bytes, offset + 8);
	entry->live_count = tq_read_u32(bytes, offset + 12);
	entry->dead_count = tq_read_u32(bytes, offset + 16);
	entry->batch_page_count = tq_read_u32(bytes, offset + 20);
	entry->summary_head_block = tq_read_u32(bytes, offset + 24);
	entry->free_lane_hint = tq_read_u16(bytes, offset + 28);
	return true;
}

bool
tq_centroid_page_init(void *page,
					  size_t page_size,
					  uint32_t dimension,
					  uint16_t centroid_capacity,
					  uint32_t next_block,
					  char *errmsg,
					  size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	size_t		required_bytes = 0;

	if (page == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: page must be non-null");
		return false;
	}

	required_bytes = tq_centroid_page_required_bytes(dimension, centroid_capacity);
	if (dimension == 0 || centroid_capacity == 0 || required_bytes > page_size)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: requested layout does not fit on the page");
		return false;
	}

	memset(page, 0, page_size);
	tq_write_page_common(bytes, TQ_PAGE_KIND_CENTROID, TQ_CENTROID_PAGE_HEADER_BYTES);
	tq_write_u32(bytes, TQ_CENTROID_DIMENSION_OFFSET, dimension);
	tq_write_u16(bytes, TQ_CENTROID_CAPACITY_OFFSET, centroid_capacity);
	tq_write_u16(bytes, TQ_CENTROID_COUNT_OFFSET, 0);
	tq_write_u32(bytes, TQ_CENTROID_NEXT_BLOCK_OFFSET, next_block);
	return true;
}

bool
tq_centroid_page_read_header(const void *page,
							 size_t page_size,
							 TqCentroidPageHeaderView *header,
							 char *errmsg,
							 size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || header == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: page and header must be non-null");
		return false;
	}

	if (!tq_centroid_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	header->dimension = tq_read_u32(bytes, TQ_CENTROID_DIMENSION_OFFSET);
	header->centroid_capacity = tq_read_u16(bytes, TQ_CENTROID_CAPACITY_OFFSET);
	header->centroid_count = tq_read_u16(bytes, TQ_CENTROID_COUNT_OFFSET);
	header->next_block = tq_read_u32(bytes, TQ_CENTROID_NEXT_BLOCK_OFFSET);
	return true;
}

bool
tq_centroid_page_set_centroid(void *page,
							  size_t page_size,
							  uint16_t index,
							  const float *values,
							  size_t value_count,
							  char *errmsg,
							  size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	uint16_t	centroid_count = 0;
	uint16_t	centroid_capacity = 0;
	uint32_t	dimension = 0;
	size_t		offset = 0;

	if (page == NULL || values == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: page and values must be non-null");
		return false;
	}

	if (!tq_centroid_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	dimension = tq_read_u32(bytes, TQ_CENTROID_DIMENSION_OFFSET);
	centroid_capacity = tq_read_u16(bytes, TQ_CENTROID_CAPACITY_OFFSET);
	if (index >= centroid_capacity)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: centroid index out of range");
		return false;
	}

	if (value_count != dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: centroid dimension does not match page layout");
		return false;
	}

	offset = tq_centroid_offset(bytes, index);
	memcpy(bytes + offset, values, sizeof(float) * (size_t) dimension);
	centroid_count = tq_read_u16(bytes, TQ_CENTROID_COUNT_OFFSET);
	if (index >= centroid_count)
		tq_write_u16(bytes, TQ_CENTROID_COUNT_OFFSET, (uint16_t) (index + 1));
	return true;
}

bool
tq_centroid_page_get_centroid(const void *page,
							  size_t page_size,
							  uint16_t index,
							  float *values,
							  size_t value_count,
							  char *errmsg,
							  size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	uint32_t	dimension = 0;
	size_t		offset = 0;

	if (page == NULL || values == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: page and values must be non-null");
		return false;
	}

	if (!tq_centroid_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	dimension = tq_read_u32(bytes, TQ_CENTROID_DIMENSION_OFFSET);
	if (index >= tq_read_u16(bytes, TQ_CENTROID_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: centroid has not been initialized");
		return false;
	}

	if (value_count < dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant centroid page: output buffer is too small");
		return false;
	}

	offset = tq_centroid_offset(bytes, index);
	memcpy(values, bytes + offset, sizeof(float) * (size_t) dimension);
	return true;
}

bool
tq_batch_summary_page_init(void *page,
						   size_t page_size,
						   uint32_t code_bytes,
						   uint16_t entry_capacity,
						   uint32_t next_block,
						   char *errmsg,
						   size_t errmsg_len)
{
	uint8_t *bytes = (uint8_t *) page;
	size_t required_bytes = 0;

	if (page == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: page must be non-null");
		return false;
	}

	required_bytes = tq_batch_summary_page_required_bytes(entry_capacity, code_bytes);
	if (code_bytes == 0 || entry_capacity == 0 || required_bytes > page_size)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: requested layout does not fit on the page");
		return false;
	}

	memset(page, 0, page_size);
	tq_write_page_common(bytes, TQ_PAGE_KIND_BATCH_SUMMARY, TQ_BATCH_SUMMARY_PAGE_HEADER_BYTES);
	tq_write_u32(bytes, TQ_BATCH_SUMMARY_CODE_BYTES_OFFSET, code_bytes);
	tq_write_u16(bytes, TQ_BATCH_SUMMARY_ENTRY_CAPACITY_OFFSET, entry_capacity);
	tq_write_u16(bytes, TQ_BATCH_SUMMARY_ENTRY_COUNT_OFFSET, 0);
	tq_write_u32(bytes, TQ_BATCH_SUMMARY_NEXT_BLOCK_OFFSET, next_block);
	return true;
}

bool
tq_batch_summary_page_read_header(const void *page,
								  size_t page_size,
								  TqBatchSummaryPageHeaderView *header,
								  char *errmsg,
								  size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || header == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: page and header must be non-null");
		return false;
	}

	if (!tq_batch_summary_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	header->code_bytes = tq_read_u32(bytes, TQ_BATCH_SUMMARY_CODE_BYTES_OFFSET);
	header->entry_capacity = tq_read_u16(bytes, TQ_BATCH_SUMMARY_ENTRY_CAPACITY_OFFSET);
	header->entry_count = tq_read_u16(bytes, TQ_BATCH_SUMMARY_ENTRY_COUNT_OFFSET);
	header->next_block = tq_read_u32(bytes, TQ_BATCH_SUMMARY_NEXT_BLOCK_OFFSET);
	return true;
}

bool
tq_batch_summary_page_set_next_block(void *page,
									 size_t page_size,
									 uint32_t next_block,
									 char *errmsg,
									 size_t errmsg_len)
{
	uint8_t *bytes = (uint8_t *) page;

	if (page == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: page must be non-null");
		return false;
	}

	if (!tq_batch_summary_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	tq_write_u32(bytes, TQ_BATCH_SUMMARY_NEXT_BLOCK_OFFSET, next_block);
	return true;
}

bool
tq_batch_summary_page_set_entry(void *page,
								size_t page_size,
								uint16_t index,
								uint32_t block_number,
								const TqBatchPageSummary *summary,
								const uint8_t *representative_code,
								size_t code_len,
								char *errmsg,
								size_t errmsg_len)
{
	uint8_t *bytes = (uint8_t *) page;
	uint32_t code_bytes = 0;
	uint16_t entry_count = 0;
	size_t offset = 0;

	if (page == NULL || summary == NULL || representative_code == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: page, summary, and code must be non-null");
		return false;
	}

	if (!tq_batch_summary_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	code_bytes = tq_read_u32(bytes, TQ_BATCH_SUMMARY_CODE_BYTES_OFFSET);
	if (code_len != (size_t) code_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: representative code length does not match layout");
		return false;
	}

	if (index >= tq_read_u16(bytes, TQ_BATCH_SUMMARY_ENTRY_CAPACITY_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: entry index out of range");
		return false;
	}

	offset = tq_batch_summary_entry_offset(bytes, index);
	tq_write_u32(bytes, offset + 0, block_number);
	tq_write_u16(bytes, offset + 4, summary->representative_lane);
	tq_write_u16(bytes, offset + 6, 0);
	tq_write_float32(bytes, offset + 8, summary->residual_radius);
	memcpy(bytes + offset + 12, representative_code, code_len);

	entry_count = tq_read_u16(bytes, TQ_BATCH_SUMMARY_ENTRY_COUNT_OFFSET);
	if (index >= entry_count)
		tq_write_u16(bytes, TQ_BATCH_SUMMARY_ENTRY_COUNT_OFFSET, (uint16_t) (index + 1));
	return true;
}

bool
tq_batch_summary_page_get_entry(const void *page,
								size_t page_size,
								uint16_t index,
								uint32_t *block_number,
								TqBatchPageSummary *summary,
								uint8_t *representative_code,
								size_t code_len,
								char *errmsg,
								size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	uint32_t code_bytes = 0;
	size_t offset = 0;

	if (page == NULL || block_number == NULL || summary == NULL || representative_code == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: page, outputs, and code buffer must be non-null");
		return false;
	}

	if (!tq_batch_summary_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (index >= tq_read_u16(bytes, TQ_BATCH_SUMMARY_ENTRY_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: entry has not been initialized");
		return false;
	}

	code_bytes = tq_read_u32(bytes, TQ_BATCH_SUMMARY_CODE_BYTES_OFFSET);
	if (code_len < (size_t) code_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch summary page: code buffer is too small");
		return false;
	}

	offset = tq_batch_summary_entry_offset(bytes, index);
	*block_number = tq_read_u32(bytes, offset + 0);
	summary->representative_lane = tq_read_u16(bytes, offset + 4);
	summary->residual_radius = tq_read_float32(bytes, offset + 8);
	memcpy(representative_code, bytes + offset + 12, (size_t) code_bytes);
	return true;
}

bool
tq_batch_page_init(void *page,
				   size_t page_size,
				   const TqBatchPageParams *params,
				   char *errmsg,
				   size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	size_t		bitmap_offset = TQ_BATCH_PAGE_HEADER_BYTES;
	size_t		total_bytes = 0;
	uint16_t	flags = 0;

	if (page == NULL || params == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and params must be non-null");
		return false;
	}

	if (params->dimension > 0)
	{
		/* SoA nibble layout */
		total_bytes = tq_batch_soa_representative_offset(params->lane_count,
														 params->dimension,
														 params->int4_attribute_count)
					  + (size_t) params->code_bytes;

		if (params->lane_count == 0 || params->code_bytes == 0
			|| total_bytes > page_size)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant batch page: requested SoA layout does not fit on the page");
			return false;
		}

		if (params->int4_attribute_count > TQ_MAX_STORED_INT4_ATTRIBUTES)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant batch page: too many int4 attributes");
			return false;
		}

		flags = tq_batch_make_flags(true, params->int4_attribute_count);

		memset(page, 0, page_size);
		tq_write_page_common(bytes, TQ_PAGE_KIND_BATCH, TQ_BATCH_PAGE_HEADER_BYTES);
		tq_write_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET, params->lane_count);
		tq_write_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET, 0);
		tq_write_u16(bytes, TQ_BATCH_LIVE_COUNT_OFFSET, 0);
		tq_write_u16(bytes, TQ_BATCH_REPRESENTATIVE_LANE_OFFSET, TQ_BATCH_PAGE_NO_REPRESENTATIVE);
		tq_write_u16(bytes, TQ_BATCH_FLAGS_OFFSET, flags);
		tq_write_u32(bytes, TQ_BATCH_LIST_ID_OFFSET, params->list_id);
		tq_write_u32(bytes, TQ_BATCH_NEXT_BLOCK_OFFSET, params->next_block);
		tq_write_u32(bytes, TQ_BATCH_CODE_BYTES_OFFSET, params->code_bytes);
		tq_write_u16(bytes, TQ_BATCH_BITMAP_OFFSET_OFFSET, (uint16_t) bitmap_offset);
		/* Repurpose tid_offset to store dimension for SoA pages */
		tq_write_u16(bytes, TQ_BATCH_TID_OFFSET_OFFSET, (uint16_t) params->dimension);
		tq_write_u16(bytes, TQ_BATCH_CODE_OFFSET_OFFSET, 0);
		tq_write_u16(bytes, TQ_BATCH_TOTAL_BYTES_OFFSET, (uint16_t) total_bytes);
		tq_write_float32(bytes, TQ_BATCH_RESIDUAL_RADIUS_OFFSET, 0.0f);
	}
	else
	{
		/* Legacy AoS interleaved layout */
		size_t	tid_offset;
		size_t	code_offset;

		if (tq_batch_required_bytes_internal(params->lane_count,
											 params->code_bytes,
											 params->int4_attribute_count) > page_size)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant batch page: requested layout does not fit on the page");
			return false;
		}

		if (params->int4_attribute_count > TQ_MAX_STORED_INT4_ATTRIBUTES)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant batch page: too many int4 attributes");
			return false;
		}

		tid_offset = bitmap_offset + tq_bitmap_bytes_for_lanes(params->lane_count);
		code_offset = tid_offset + (size_t) TQ_TID_STORAGE_BYTES
			+ ((size_t) params->int4_attribute_count * sizeof(int32_t));
		total_bytes = tq_batch_required_bytes_internal(params->lane_count,
													   params->code_bytes,
													   params->int4_attribute_count);
		flags = tq_batch_make_flags(false, params->int4_attribute_count);

		memset(page, 0, page_size);
		tq_write_page_common(bytes, TQ_PAGE_KIND_BATCH, TQ_BATCH_PAGE_HEADER_BYTES);
		tq_write_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET, params->lane_count);
		tq_write_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET, 0);
		tq_write_u16(bytes, TQ_BATCH_LIVE_COUNT_OFFSET, 0);
		tq_write_u16(bytes, TQ_BATCH_REPRESENTATIVE_LANE_OFFSET, TQ_BATCH_PAGE_NO_REPRESENTATIVE);
		tq_write_u16(bytes, TQ_BATCH_FLAGS_OFFSET, flags);
		tq_write_u32(bytes, TQ_BATCH_LIST_ID_OFFSET, params->list_id);
		tq_write_u32(bytes, TQ_BATCH_NEXT_BLOCK_OFFSET, params->next_block);
		tq_write_u32(bytes, TQ_BATCH_CODE_BYTES_OFFSET, params->code_bytes);
		tq_write_u16(bytes, TQ_BATCH_BITMAP_OFFSET_OFFSET, (uint16_t) bitmap_offset);
		tq_write_u16(bytes, TQ_BATCH_TID_OFFSET_OFFSET, (uint16_t) tid_offset);
		tq_write_u16(bytes, TQ_BATCH_CODE_OFFSET_OFFSET, (uint16_t) code_offset);
		tq_write_u16(bytes, TQ_BATCH_TOTAL_BYTES_OFFSET, (uint16_t) total_bytes);
		tq_write_float32(bytes, TQ_BATCH_RESIDUAL_RADIUS_OFFSET, 0.0f);
	}

	return true;
}

bool
tq_batch_page_used_bytes(const void *page,
						 size_t page_size,
						 size_t *used_bytes,
						 char *errmsg,
						 size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || used_bytes == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and used-bytes output must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (tq_batch_is_soa_page(bytes))
	{
		/*
		 * SoA pages have a fixed total footprint determined at init time.
		 * All regions (TID array, gamma array, nibble block, representative)
		 * are pre-allocated for lane_count slots.
		 */
		*used_bytes = (size_t) tq_read_u16(bytes, TQ_BATCH_TOTAL_BYTES_OFFSET);
	}
	else
	{
		size_t		entry_offset;
		uint16_t	occupied_count;
		size_t		entry_stride;

		entry_offset = (size_t) tq_read_u16(bytes, TQ_BATCH_TID_OFFSET_OFFSET);
		occupied_count = tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET);
		entry_stride = tq_batch_entry_stride_from_layout(bytes);

		*used_bytes = entry_offset + (entry_stride * (size_t) occupied_count);
	}
	return true;
}

bool
tq_batch_page_set_next_block(void *page,
							 size_t page_size,
							 uint32_t next_block,
							 char *errmsg,
							 size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;

	if (page == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	tq_write_u32(bytes, TQ_BATCH_NEXT_BLOCK_OFFSET, next_block);
	return true;
}

bool
tq_batch_page_has_capacity(const void *page,
						   size_t page_size,
						   bool *has_capacity,
						   char *errmsg,
						   size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || has_capacity == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and capacity output must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	*has_capacity = tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET)
		< tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);
	return true;
}

bool
tq_batch_page_should_reclaim(const void *page,
							 size_t page_size,
							 bool *should_reclaim,
							 char *errmsg,
							 size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	uint16_t	occupied_count = 0;
	uint16_t	live_count = 0;

	if (page == NULL || should_reclaim == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and reclaim output must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	occupied_count = tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET);
	live_count = tq_read_u16(bytes, TQ_BATCH_LIVE_COUNT_OFFSET);
	*should_reclaim = occupied_count > 0 && live_count == 0;
	return true;
}

bool
tq_batch_page_read_header(const void *page,
						  size_t page_size,
						  TqBatchPageHeaderView *header,
						  char *errmsg,
						  size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || header == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and header must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	header->lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);
	header->occupied_count = tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET);
	header->live_count = tq_read_u16(bytes, TQ_BATCH_LIVE_COUNT_OFFSET);
	header->representative_lane = tq_read_u16(bytes, TQ_BATCH_REPRESENTATIVE_LANE_OFFSET);
	header->code_bytes = tq_read_u32(bytes, TQ_BATCH_CODE_BYTES_OFFSET);
	header->list_id = tq_read_u32(bytes, TQ_BATCH_LIST_ID_OFFSET);
	header->next_block = tq_read_u32(bytes, TQ_BATCH_NEXT_BLOCK_OFFSET);
	header->residual_radius = tq_read_float32(bytes, TQ_BATCH_RESIDUAL_RADIUS_OFFSET);
	header->flags = tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET);
	return true;
}

bool
tq_batch_page_set_summary(void *page,
						  size_t page_size,
						  const TqBatchPageSummary *summary,
						  char *errmsg,
						  size_t errmsg_len)
{
	uint8_t *bytes = (uint8_t *) page;

	if (page == NULL || summary == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and summary must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (summary->representative_lane != TQ_BATCH_PAGE_NO_REPRESENTATIVE
		&& summary->representative_lane >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: summary representative lane exceeds occupied lanes");
		return false;
	}

	if (summary->residual_radius < 0.0f)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: summary residual radius must be non-negative");
		return false;
	}

	tq_write_u16(bytes, TQ_BATCH_REPRESENTATIVE_LANE_OFFSET, summary->representative_lane);
	tq_write_float32(bytes, TQ_BATCH_RESIDUAL_RADIUS_OFFSET, summary->residual_radius);
	return true;
}

bool
tq_batch_page_get_summary(const void *page,
						  size_t page_size,
						  TqBatchPageSummary *summary,
						  char *errmsg,
						  size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || summary == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and summary output must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	summary->representative_lane = tq_read_u16(bytes, TQ_BATCH_REPRESENTATIVE_LANE_OFFSET);
	summary->residual_radius = tq_read_float32(bytes, TQ_BATCH_RESIDUAL_RADIUS_OFFSET);
	return true;
}

bool
tq_batch_page_append_lane(void *page,
						  size_t page_size,
						  const TqTid *tid,
						  uint16_t *lane_index,
						  char *errmsg,
						  size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	uint16_t	occupied_count = 0;
	uint16_t	lane_count = 0;

	if (page == NULL || tid == NULL || lane_index == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page, tid, and lane index must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	occupied_count = tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET);
	lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);

	if (occupied_count >= lane_count)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: no free lanes remain");
		return false;
	}

	if (tq_batch_is_soa_page(bytes))
	{
		/* SoA layout: write TID to contiguous TID array */
		size_t	tid_off = tq_batch_soa_tid_array_offset(lane_count)
						  + (size_t) occupied_count * TQ_TID_STORAGE_BYTES;

		tq_write_u32(bytes, tid_off, tid->block_number);
		tq_write_u16(bytes, tid_off + 4, tid->offset_number);
	}
	else
	{
		tq_write_tid(bytes, occupied_count, tid);
	}

	tq_batch_set_live(bytes, occupied_count, true);
	tq_write_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET, (uint16_t) (occupied_count + 1));
	tq_write_u16(bytes, TQ_BATCH_LIVE_COUNT_OFFSET,
				 (uint16_t) (tq_read_u16(bytes, TQ_BATCH_LIVE_COUNT_OFFSET) + 1));
	*lane_index = occupied_count;
	return true;
}

bool
tq_batch_page_get_tid(const void *page,
					  size_t page_size,
					  uint16_t lane_index,
					  TqTid *tid,
					  char *errmsg,
					  size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || tid == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and tid must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane index is not occupied");
		return false;
	}

	if (tq_batch_is_soa_page(bytes))
	{
		uint16_t	lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);
		size_t		tid_off = tq_batch_soa_tid_array_offset(lane_count)
							  + (size_t) lane_index * TQ_TID_STORAGE_BYTES;

		tid->block_number = tq_read_u32(bytes, tid_off);
		tid->offset_number = tq_read_u16(bytes, tid_off + 4);
	}
	else
	{
		tq_read_tid(bytes, lane_index, tid);
	}

	return true;
}

bool
tq_batch_page_get_int4_attribute_count(const void *page,
									   size_t page_size,
									   uint16_t *attribute_count,
									   char *errmsg,
									   size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || attribute_count == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and attribute count output must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	*attribute_count = tq_batch_flag_int4_attribute_count(
		tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET));
	return true;
}

bool
tq_batch_page_set_int4_attribute(void *page,
								 size_t page_size,
								 uint16_t lane_index,
								 uint16_t attribute_index,
								 int32_t value,
								 char *errmsg,
								 size_t errmsg_len)
{
	uint8_t *bytes = (uint8_t *) page;
	uint16_t attribute_count = 0;
	size_t offset = 0;

	if (page == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page must be non-null");
		return false;
	}

	if (!tq_batch_page_get_int4_attribute_count(page, page_size, &attribute_count,
												errmsg, errmsg_len))
		return false;

	if (attribute_count == 0 || attribute_index >= attribute_count)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: requested int4 attribute is not available on this page");
		return false;
	}

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane index is not occupied");
		return false;
	}

	if (tq_batch_is_soa_page(bytes))
	{
		uint16_t lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);

		offset = tq_batch_soa_int4_attribute_offset(lane_count, lane_index, attribute_index);
	}
	else
		offset = tq_batch_int4_attribute_offset(bytes, lane_index, attribute_index);

	tq_write_u32(bytes, offset, (uint32_t) value);
	return true;
}

bool
tq_batch_page_get_int4_attribute(const void *page,
								 size_t page_size,
								 uint16_t lane_index,
								 uint16_t attribute_index,
								 int32_t *value,
								 char *errmsg,
								 size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	uint16_t attribute_count = 0;
	size_t offset = 0;

	if (page == NULL || value == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and attribute output must be non-null");
		return false;
	}

	if (!tq_batch_page_get_int4_attribute_count(page, page_size, &attribute_count,
												errmsg, errmsg_len))
		return false;

	if (attribute_count == 0 || attribute_index >= attribute_count)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: requested int4 attribute is not available on this page");
		return false;
	}

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane index is not occupied");
		return false;
	}

	if (tq_batch_is_soa_page(bytes))
	{
		uint16_t lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);

		offset = tq_batch_soa_int4_attribute_offset(lane_count, lane_index, attribute_index);
	}
	else
		offset = tq_batch_int4_attribute_offset(bytes, lane_index, attribute_index);

	*value = (int32_t) tq_read_u32(bytes, offset);
	return true;
}

bool
tq_batch_page_has_filter_int4(const void *page,
							  size_t page_size,
							  bool *has_filter,
							  char *errmsg,
							  size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || has_filter == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and filter flag output must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	*has_filter = tq_batch_flag_int4_attribute_count(
		tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET)) > 0;
	return true;
}

bool
tq_batch_page_set_filter_int4(void *page,
							  size_t page_size,
							  uint16_t lane_index,
							  int32_t filter_value,
							  char *errmsg,
							  size_t errmsg_len)
{
	return tq_batch_page_set_int4_attribute(page, page_size, lane_index, 0,
												filter_value, errmsg, errmsg_len);
}

bool
tq_batch_page_get_filter_int4(const void *page,
							  size_t page_size,
							  uint16_t lane_index,
							  int32_t *filter_value,
							  char *errmsg,
							  size_t errmsg_len)
{
	return tq_batch_page_get_int4_attribute(page, page_size, lane_index, 0,
												filter_value, errmsg, errmsg_len);
}

bool
tq_batch_page_set_code(void *page,
					   size_t page_size,
					   uint16_t lane_index,
					   const uint8_t *code,
					   size_t code_len,
					   char *errmsg,
					   size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	size_t		stored_len = 0;
	size_t		offset = 0;

	if (page == NULL || code == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and code must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (tq_batch_is_soa_page(bytes))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: per-lane set_code is not available on SoA pages; use set_nibble_and_gamma or set_representative_code");
		return false;
	}

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane index is not occupied");
		return false;
	}

	stored_len = (size_t) tq_read_u32(bytes, TQ_BATCH_CODE_BYTES_OFFSET);
	if (code_len != stored_len)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: code length does not match page layout");
		return false;
	}

	offset = tq_batch_code_offset(bytes, lane_index);
	memcpy(bytes + offset, code, stored_len);
	return true;
}

bool
tq_batch_page_get_code(const void *page,
					   size_t page_size,
					   uint16_t lane_index,
					   uint8_t *code,
					   size_t code_len,
					   char *errmsg,
					   size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	size_t		stored_len = 0;
	size_t		offset = 0;

	if (page == NULL || code == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and code must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (tq_batch_is_soa_page(bytes))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: per-lane get_code is not available on SoA pages; use nibble/gamma or representative accessors");
		return false;
	}

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane index is not occupied");
		return false;
	}

	stored_len = (size_t) tq_read_u32(bytes, TQ_BATCH_CODE_BYTES_OFFSET);
	if (code_len < stored_len)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: output code buffer is too small");
		return false;
	}

	offset = tq_batch_code_offset(bytes, lane_index);
	memcpy(code, bytes + offset, stored_len);
	return true;
}

bool
tq_batch_page_code_view(const void *page,
						size_t page_size,
						uint16_t lane_index,
						const uint8_t **code,
						size_t *code_len,
						char *errmsg,
						size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	size_t stored_len = 0;
	size_t offset = 0;

	if (page == NULL || code == NULL || code_len == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page, code view, and code length must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane index is not occupied");
		return false;
	}

	if (tq_batch_is_soa_page(bytes))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: per-lane code view is not available on SoA pages; use nibble/gamma or representative accessors");
		return false;
	}

	stored_len = (size_t) tq_read_u32(bytes, TQ_BATCH_CODE_BYTES_OFFSET);
	offset = tq_batch_code_offset(bytes, lane_index);
	*code = bytes + offset;
	*code_len = stored_len;
	return true;
}

bool
tq_batch_page_mark_dead(void *page,
						size_t page_size,
						uint16_t lane_index,
						char *errmsg,
						size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	uint16_t	live_count = 0;

	if (page == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane index is not occupied");
		return false;
	}

	if (!tq_batch_lane_is_live(bytes, lane_index))
		return true;

	tq_batch_set_live(bytes, lane_index, false);
	live_count = tq_read_u16(bytes, TQ_BATCH_LIVE_COUNT_OFFSET);
	tq_write_u16(bytes, TQ_BATCH_LIVE_COUNT_OFFSET, (uint16_t) (live_count - 1));
	return true;
}

bool
tq_batch_page_compact(void *page,
					  size_t page_size,
					  char *errmsg,
					  size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	uint16_t	occupied_count = 0;
	uint16_t	write_lane = 0;
	uint16_t	read_lane = 0;

	if (page == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	occupied_count = tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET);

	if (tq_batch_is_soa_page(bytes))
	{
		uint16_t	lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);
		uint16_t	dimension = tq_read_u16(bytes, TQ_BATCH_TID_OFFSET_OFFSET);
		uint16_t	int4_attribute_count = tq_batch_flag_int4_attribute_count(
			tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET));
		size_t		tid_base = tq_batch_soa_tid_array_offset(lane_count);
		size_t		filter_base = tq_batch_soa_filter_array_offset(lane_count);
		size_t		gamma_base = tq_batch_soa_gamma_array_offset(lane_count, int4_attribute_count);
		size_t		nibble_base = tq_batch_soa_nibble_block_offset(lane_count, int4_attribute_count);
		uint32_t	d;

		for (read_lane = 0; read_lane < occupied_count; read_lane++)
		{
			if (tq_batch_lane_is_live(bytes, read_lane))
			{
				if (write_lane != read_lane)
				{
					/* Move TID */
					memcpy(bytes + tid_base + (size_t) write_lane * TQ_TID_STORAGE_BYTES,
						   bytes + tid_base + (size_t) read_lane * TQ_TID_STORAGE_BYTES,
						   TQ_TID_STORAGE_BYTES);

					if (int4_attribute_count > 0)
					{
						uint16_t attribute_index = 0;

						for (attribute_index = 0; attribute_index < int4_attribute_count; attribute_index++)
							memcpy(bytes + filter_base
								   + ((((size_t) attribute_index * (size_t) lane_count)
									   + (size_t) write_lane) * sizeof(int32_t)),
								   bytes + filter_base
								   + ((((size_t) attribute_index * (size_t) lane_count)
									   + (size_t) read_lane) * sizeof(int32_t)),
								   sizeof(int32_t));
					}

					/* Move gamma */
					memcpy(bytes + gamma_base + (size_t) write_lane * sizeof(float),
						   bytes + gamma_base + (size_t) read_lane * sizeof(float),
						   sizeof(float));

					/* Move nibbles: extract from source slot, write to dest slot (4-bit packed) */
					{
						size_t	pair_cols = ((size_t) lane_count + 1u) / 2u;
						size_t	src_col = (size_t) read_lane / 2u;
						bool	src_high = (read_lane % 2u) != 0;
						size_t	dst_col = (size_t) write_lane / 2u;
						bool	dst_high = (write_lane % 2u) != 0;

						for (d = 0; d < dimension; d++)
						{
							size_t	src_off = nibble_base + (size_t) d * pair_cols + src_col;
							size_t	dst_off = nibble_base + (size_t) d * pair_cols + dst_col;
							uint8_t	nib = src_high
								? (bytes[src_off] >> 4u) & 0x0Fu
								: bytes[src_off] & 0x0Fu;

							if (dst_high)
								bytes[dst_off] = (bytes[dst_off] & 0x0Fu) | (nib << 4u);
							else
								bytes[dst_off] = (bytes[dst_off] & 0xF0u) | nib;
						}
					}
				}
				tq_batch_set_live(bytes, write_lane, true);
				write_lane++;
			}
		}

		while (write_lane < occupied_count)
		{
			tq_batch_set_live(bytes, write_lane, false);
			write_lane++;
		}
	}
	else
	{
		size_t		code_bytes;
		uint8_t    *scratch = NULL;

		code_bytes = (size_t) tq_read_u32(bytes, TQ_BATCH_CODE_BYTES_OFFSET);
		scratch = code_bytes > 0 ?
#ifdef TQ_UNIT_TEST
			(uint8_t *) malloc(code_bytes) :
#else
			(uint8_t *) palloc(code_bytes) :
#endif
			NULL;

		for (read_lane = 0; read_lane < occupied_count; read_lane++)
		{
			if (tq_batch_lane_is_live(bytes, read_lane))
			{
				if (write_lane != read_lane)
				{
					TqTid tid;
					uint16_t attribute_count = tq_batch_flag_int4_attribute_count(
						tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET));
					uint16_t attribute_index = 0;

					memset(&tid, 0, sizeof(tid));
					tq_read_tid(bytes, read_lane, &tid);
					tq_write_tid(bytes, write_lane, &tid);
					for (attribute_index = 0; attribute_index < attribute_count; attribute_index++)
					{
						uint32_t value = tq_read_u32(bytes,
													 tq_batch_int4_attribute_offset(bytes,
																				 read_lane,
																				 attribute_index));

						tq_write_u32(bytes,
									 tq_batch_int4_attribute_offset(bytes,
																	write_lane,
																	attribute_index),
									 value);
					}
					if (code_bytes > 0)
					{
						memcpy(scratch, bytes + tq_batch_code_offset(bytes, read_lane), code_bytes);
						memcpy(bytes + tq_batch_code_offset(bytes, write_lane), scratch, code_bytes);
					}
				}
				tq_batch_set_live(bytes, write_lane, true);
				write_lane++;
			}
		}

		while (write_lane < occupied_count)
		{
			tq_batch_set_live(bytes, write_lane, false);
			write_lane++;
		}

		if (scratch != NULL)
		{
#ifdef TQ_UNIT_TEST
			free(scratch);
#else
			pfree(scratch);
#endif
		}
	}

	tq_write_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET, tq_read_u16(bytes, TQ_BATCH_LIVE_COUNT_OFFSET));
	tq_write_u16(bytes, TQ_BATCH_REPRESENTATIVE_LANE_OFFSET, TQ_BATCH_PAGE_NO_REPRESENTATIVE);
	tq_write_float32(bytes, TQ_BATCH_RESIDUAL_RADIUS_OFFSET, 0.0f);
	return true;
}

bool
tq_batch_page_is_live(const void *page,
					  size_t page_size,
					  uint16_t lane_index,
					  bool *is_live,
					  char *errmsg,
					  size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || is_live == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and live output must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane index exceeds lane count");
		return false;
	}

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		*is_live = false;
		return true;
	}

	*is_live = tq_batch_lane_is_live(bytes, lane_index);
	return true;
}

bool
tq_batch_page_next_live_lane(const void *page,
							 size_t page_size,
							 int start_lane,
							 uint16_t *lane_index,
							 char *errmsg,
							 size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	uint16_t	occupied_count = 0;
	int			candidate = 0;

	if (page == NULL || lane_index == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and lane index must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	occupied_count = tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET);

	for (candidate = start_lane + 1; candidate < (int) occupied_count; candidate++)
	{
		if (tq_batch_lane_is_live(bytes, (uint16_t) candidate))
		{
			*lane_index = (uint16_t) candidate;
			return true;
		}
	}

	return false;
}

bool
tq_batch_page_is_soa(const void *page, size_t page_size)
{
	const uint8_t *bytes = (const uint8_t *) page;

	if (page == NULL || page_size < TQ_BATCH_PAGE_HEADER_BYTES)
		return false;

	if (tq_read_u32(bytes, TQ_PAGE_MAGIC_OFFSET) != TQ_PAGE_MAGIC)
		return false;

	if (tq_read_u16(bytes, TQ_PAGE_KIND_OFFSET) != (uint16_t) TQ_PAGE_KIND_BATCH)
		return false;

	return tq_batch_is_soa_page(bytes);
}

size_t
tq_batch_page_soa_required_bytes(uint16_t lane_count, uint32_t dimension,
								 uint32_t representative_code_bytes)
{
	return tq_batch_page_soa_required_bytes_with_filter(lane_count, dimension,
														representative_code_bytes,
														false);
}

size_t
tq_batch_page_soa_required_bytes_with_filter(uint16_t lane_count, uint32_t dimension,
											 uint32_t representative_code_bytes,
											 bool has_int4_filter)
{
	return tq_batch_page_soa_required_bytes_with_int4_attributes(
		lane_count,
		dimension,
		representative_code_bytes,
		has_int4_filter ? 1 : 0);
}

size_t
tq_batch_page_soa_required_bytes_with_int4_attributes(uint16_t lane_count,
														 uint32_t dimension,
														 uint32_t representative_code_bytes,
														 uint16_t int4_attribute_count)
{
	return tq_batch_soa_required_bytes_internal(lane_count, dimension,
												representative_code_bytes,
												int4_attribute_count);
}

bool
tq_batch_page_can_fit_soa(size_t page_size, uint16_t lane_count,
						  uint32_t dimension, uint32_t representative_code_bytes)
{
	return tq_batch_page_can_fit_soa_with_filter(page_size, lane_count, dimension,
												 representative_code_bytes, false);
}

bool
tq_batch_page_can_fit_soa_with_filter(size_t page_size, uint16_t lane_count,
									  uint32_t dimension,
									  uint32_t representative_code_bytes,
									  bool has_int4_filter)
{
	return tq_batch_page_can_fit_soa_with_int4_attributes(
		page_size,
		lane_count,
		dimension,
		representative_code_bytes,
		has_int4_filter ? 1 : 0);
}

bool
tq_batch_page_can_fit_soa_with_int4_attributes(size_t page_size,
												   uint16_t lane_count,
												   uint32_t dimension,
												   uint32_t representative_code_bytes,
												   uint16_t int4_attribute_count)
{
	if (lane_count == 0 || dimension == 0 || representative_code_bytes == 0)
		return false;

	return tq_batch_page_soa_required_bytes_with_int4_attributes(lane_count,
																dimension,
																representative_code_bytes,
																int4_attribute_count) <= page_size;
}

bool
tq_batch_page_set_nibble_and_gamma(void *page, size_t page_size,
								   uint16_t lane_index, const uint8_t *nibbles,
								   uint32_t dimension, float gamma,
								   char *errmsg, size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	uint16_t	lane_count;
	uint16_t	stored_dimension;
	size_t		nibble_base;
	size_t		gamma_base;
	uint32_t	d;

	if (page == NULL || nibbles == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and nibbles must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (!tq_batch_is_soa_page(bytes))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: set_nibble_and_gamma requires SoA page");
		return false;
	}

	lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);
	stored_dimension = tq_read_u16(bytes, TQ_BATCH_TID_OFFSET_OFFSET);

	if (dimension != (uint32_t) stored_dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: nibble dimension does not match SoA page layout");
		return false;
	}

	if (lane_index >= tq_read_u16(bytes, TQ_BATCH_OCCUPIED_COUNT_OFFSET))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: lane index is not occupied");
		return false;
	}

	nibble_base = tq_batch_soa_nibble_block_offset(lane_count,
												   tq_batch_flag_int4_attribute_count(
													   tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET)));
	{
		size_t	pair_cols = ((size_t) lane_count + 1u) / 2u;
		size_t	byte_col = (size_t) lane_index / 2u;
		bool	is_high = (lane_index % 2u) != 0;

		for (d = 0; d < dimension; d++)
		{
			size_t	off = nibble_base + (size_t) d * pair_cols + byte_col;
			uint8_t	nib = nibbles[d] & 0x0Fu;

			if (is_high)
				bytes[off] = (bytes[off] & 0x0Fu) | (nib << 4u);
			else
				bytes[off] = (bytes[off] & 0xF0u) | nib;
		}
	}

	gamma_base = tq_batch_soa_gamma_array_offset(lane_count,
												 tq_batch_flag_int4_attribute_count(
													 tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET)));
	tq_write_float32(bytes, gamma_base + (size_t) lane_index * sizeof(float), gamma);

	return true;
}

bool
tq_batch_page_get_nibble_ptr(const void *page, size_t page_size,
							 const uint8_t **nibbles, uint32_t *dimension,
							 uint16_t *lane_count, char *errmsg, size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	uint16_t	lc;

	if (page == NULL || nibbles == NULL || dimension == NULL || lane_count == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and output pointers must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (!tq_batch_is_soa_page(bytes))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: get_nibble_ptr requires SoA page");
		return false;
	}

	lc = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);
	*dimension = (uint32_t) tq_read_u16(bytes, TQ_BATCH_TID_OFFSET_OFFSET);
	*lane_count = lc;
	*nibbles = bytes + tq_batch_soa_nibble_block_offset(lc,
													 tq_batch_flag_int4_attribute_count(
														 tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET)));

	return true;
}

bool
tq_batch_page_get_gamma_ptr(const void *page, size_t page_size,
							const float **gammas, char *errmsg, size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	uint16_t	lane_count;

	if (page == NULL || gammas == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and gammas pointer must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (!tq_batch_is_soa_page(bytes))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: get_gamma_ptr requires SoA page");
		return false;
	}

	lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);
	*gammas = (const float *) (bytes + tq_batch_soa_gamma_array_offset(lane_count,
														tq_batch_flag_int4_attribute_count(
															tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET))));

	return true;
}

bool
tq_batch_page_set_representative_code(void *page, size_t page_size,
									  const uint8_t *code, size_t code_len,
									  char *errmsg, size_t errmsg_len)
{
	uint8_t    *bytes = (uint8_t *) page;
	uint16_t	lane_count;
	uint16_t	dimension;
	uint32_t	stored_code_bytes;
	size_t		rep_offset;

	if (page == NULL || code == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page and code must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (!tq_batch_is_soa_page(bytes))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: set_representative_code requires SoA page");
		return false;
	}

	lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);
	dimension = tq_read_u16(bytes, TQ_BATCH_TID_OFFSET_OFFSET);
	stored_code_bytes = tq_read_u32(bytes, TQ_BATCH_CODE_BYTES_OFFSET);

	if (code_len != (size_t) stored_code_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: representative code length does not match page layout");
		return false;
	}

	rep_offset = tq_batch_soa_representative_offset(lane_count, (uint32_t) dimension,
													tq_batch_flag_int4_attribute_count(
														tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET)));
	memcpy(bytes + rep_offset, code, code_len);
	return true;
}

bool
tq_batch_page_get_representative_code_view(const void *page, size_t page_size,
										   const uint8_t **code, size_t *code_len,
										   char *errmsg, size_t errmsg_len)
{
	const uint8_t *bytes = (const uint8_t *) page;
	uint16_t	lane_count;
	uint16_t	dimension;
	uint32_t	stored_code_bytes;
	size_t		rep_offset;

	if (page == NULL || code == NULL || code_len == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: page, code, and code_len must be non-null");
		return false;
	}

	if (!tq_batch_validate_header(bytes, page_size, errmsg, errmsg_len))
		return false;

	if (!tq_batch_is_soa_page(bytes))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant batch page: get_representative_code_view requires SoA page");
		return false;
	}

	lane_count = tq_read_u16(bytes, TQ_BATCH_LANE_COUNT_OFFSET);
	dimension = tq_read_u16(bytes, TQ_BATCH_TID_OFFSET_OFFSET);
	stored_code_bytes = tq_read_u32(bytes, TQ_BATCH_CODE_BYTES_OFFSET);

	rep_offset = tq_batch_soa_representative_offset(lane_count, (uint32_t) dimension,
													tq_batch_flag_int4_attribute_count(
														tq_read_u16(bytes, TQ_BATCH_FLAGS_OFFSET)));
	*code = bytes + rep_offset;
	*code_len = (size_t) stored_code_bytes;
	return true;
}
