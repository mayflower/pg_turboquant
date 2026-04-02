#ifndef TQ_PAGE_H
#define TQ_PAGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/tq_options.h"
#include "src/tq_transform.h"

#define TQ_PAGE_MAGIC UINT32_C(0x54515047)
#define TQ_PAGE_FORMAT_VERSION 12
#define TQ_INVALID_BLOCK_NUMBER UINT32_MAX
#define TQ_DETACHED_FREE_LIST_ID UINT32_MAX
#define TQ_BATCH_PAGE_NO_REPRESENTATIVE UINT16_MAX

#define TQ_ALGORITHM_VERSION 3
#define TQ_QUANTIZER_VERSION 2
#define TQ_RESIDUAL_SKETCH_VERSION 2
#define TQ_ESTIMATOR_VERSION 2

typedef enum TqPageKind
{
	TQ_PAGE_KIND_META = 1,
	TQ_PAGE_KIND_LIST_DIRECTORY = 2,
	TQ_PAGE_KIND_BATCH = 3,
	TQ_PAGE_KIND_CENTROID = 4,
	TQ_PAGE_KIND_BATCH_SUMMARY = 5
} TqPageKind;

typedef enum TqDistanceKind
{
	TQ_DISTANCE_COSINE = 1,
	TQ_DISTANCE_IP = 2,
	TQ_DISTANCE_L2 = 3
} TqDistanceKind;

typedef struct TqTid
{
	uint32_t	block_number;
	uint16_t	offset_number;
} TqTid;

typedef struct TqMetaPageFields
{
	uint32_t		dimension;
	uint32_t		transform_output_dimension;
	TqCodecKind		codec;
	TqDistanceKind	distance;
	uint16_t		bits;
	uint16_t		lane_count;
	TqTransformKind	transform;
	uint16_t		transform_version;
	bool			normalized;
	uint32_t		list_count;
	uint32_t		directory_root_block;
	uint32_t		centroid_root_block;
	uint64_t		transform_seed;
	uint32_t		router_seed;
	uint32_t		router_sample_count;
	uint32_t		router_max_iterations;
	uint32_t		router_completed_iterations;
	uint32_t		router_trained_vector_count;
	TqRouterAlgorithmKind router_algorithm;
	uint32_t		router_restart_count;
	uint32_t		router_selected_restart;
	float			router_mean_distortion;
	float			router_max_list_over_avg;
	float			router_coeff_var;
	float			router_balance_penalty;
	float			router_selection_score;
	uint16_t		algorithm_version;
	uint16_t		quantizer_version;
	uint16_t		residual_sketch_version;
	uint16_t		residual_bits_per_dimension;
	uint32_t		residual_sketch_dimension;
	uint16_t		estimator_version;
} TqMetaPageFields;

typedef struct TqListDirEntry
{
	uint32_t	list_id;
	uint32_t	head_block;
	uint32_t	tail_block;
	uint32_t	live_count;
	uint32_t	dead_count;
	uint32_t	batch_page_count;
	uint32_t	summary_head_block;
	uint16_t	free_lane_hint;
} TqListDirEntry;

typedef struct TqListDirPageHeaderView
{
	uint16_t	entry_capacity;
	uint16_t	entry_count;
	uint32_t	next_block;
} TqListDirPageHeaderView;

typedef struct TqBatchPageParams
{
	uint16_t	lane_count;
	uint32_t	code_bytes;
	uint32_t	list_id;
	uint32_t	next_block;
	uint32_t	dimension;		/* >0 selects SoA nibble layout; 0 uses legacy AoS */
} TqBatchPageParams;

typedef struct TqBatchPageHeaderView
{
	uint16_t	lane_count;
	uint16_t	occupied_count;
	uint16_t	live_count;
	uint16_t	representative_lane;
	uint32_t	code_bytes;
	uint32_t	list_id;
	uint32_t	next_block;
	float		residual_radius;
	uint16_t	flags;
} TqBatchPageHeaderView;

typedef struct TqBatchPageSummary
{
	uint16_t	representative_lane;
	float		residual_radius;
} TqBatchPageSummary;

typedef struct TqCentroidPageHeaderView
{
	uint32_t	dimension;
	uint16_t	centroid_capacity;
	uint16_t	centroid_count;
	uint32_t	next_block;
} TqCentroidPageHeaderView;

typedef struct TqBatchSummaryPageHeaderView
{
	uint32_t	code_bytes;
	uint16_t	entry_capacity;
	uint16_t	entry_count;
	uint32_t	next_block;
} TqBatchSummaryPageHeaderView;

extern size_t tq_bitmap_bytes_for_lanes(uint16_t lane_count);
extern size_t tq_batch_page_required_bytes(uint16_t lane_count, uint32_t code_bytes);
extern bool tq_batch_page_can_fit(size_t page_size, uint16_t lane_count, uint32_t code_bytes);
extern bool tq_batch_page_used_bytes(const void *page,
									 size_t page_size,
									 size_t *used_bytes,
									 char *errmsg,
									 size_t errmsg_len);
extern uint16_t tq_list_dir_page_capacity(size_t page_size);
extern size_t tq_centroid_page_required_bytes(uint32_t dimension, uint16_t centroid_capacity);
extern uint16_t tq_centroid_page_capacity(size_t page_size, uint32_t dimension);
extern size_t tq_batch_summary_page_required_bytes(uint16_t entry_capacity, uint32_t code_bytes);
extern uint16_t tq_batch_summary_page_capacity(size_t page_size, uint32_t code_bytes);

extern bool tq_meta_page_init(void *page,
							  size_t page_size,
							  const TqMetaPageFields *fields,
							  char *errmsg,
							  size_t errmsg_len);
extern bool tq_page_read_kind(const void *page,
							  size_t page_size,
							  TqPageKind *kind,
							  char *errmsg,
							  size_t errmsg_len);
extern bool tq_meta_page_read(const void *page,
							  size_t page_size,
							  TqMetaPageFields *fields,
							  char *errmsg,
							  size_t errmsg_len);

extern bool tq_list_dir_page_init(void *page,
								  size_t page_size,
								  uint16_t entry_capacity,
								  uint32_t next_block,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_list_dir_page_read_header(const void *page,
										 size_t page_size,
										 TqListDirPageHeaderView *header,
										 char *errmsg,
										 size_t errmsg_len);
extern bool tq_list_dir_page_set_entry(void *page,
									   size_t page_size,
									   uint16_t index,
									   const TqListDirEntry *entry,
									   char *errmsg,
									   size_t errmsg_len);
extern bool tq_list_dir_page_get_entry(const void *page,
									   size_t page_size,
									   uint16_t index,
									   TqListDirEntry *entry,
									   char *errmsg,
									   size_t errmsg_len);
extern bool tq_centroid_page_init(void *page,
								  size_t page_size,
								  uint32_t dimension,
								  uint16_t centroid_capacity,
								  uint32_t next_block,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_centroid_page_read_header(const void *page,
										 size_t page_size,
										 TqCentroidPageHeaderView *header,
										 char *errmsg,
										 size_t errmsg_len);
extern bool tq_centroid_page_set_centroid(void *page,
										  size_t page_size,
										  uint16_t index,
										  const float *values,
										  size_t value_count,
										  char *errmsg,
										  size_t errmsg_len);
extern bool tq_centroid_page_get_centroid(const void *page,
										  size_t page_size,
										  uint16_t index,
										  float *values,
										  size_t value_count,
										  char *errmsg,
										  size_t errmsg_len);
extern bool tq_batch_summary_page_init(void *page,
									   size_t page_size,
									   uint32_t code_bytes,
									   uint16_t entry_capacity,
									   uint32_t next_block,
									   char *errmsg,
									   size_t errmsg_len);
extern bool tq_batch_summary_page_read_header(const void *page,
											  size_t page_size,
											  TqBatchSummaryPageHeaderView *header,
											  char *errmsg,
											  size_t errmsg_len);
extern bool tq_batch_summary_page_set_next_block(void *page,
												 size_t page_size,
												 uint32_t next_block,
												 char *errmsg,
												 size_t errmsg_len);
extern bool tq_batch_summary_page_set_entry(void *page,
											size_t page_size,
											uint16_t index,
											uint32_t block_number,
											const TqBatchPageSummary *summary,
											const uint8_t *representative_code,
											size_t code_len,
											char *errmsg,
											size_t errmsg_len);
extern bool tq_batch_summary_page_get_entry(const void *page,
											size_t page_size,
											uint16_t index,
											uint32_t *block_number,
											TqBatchPageSummary *summary,
											uint8_t *representative_code,
											size_t code_len,
											char *errmsg,
											size_t errmsg_len);

extern bool tq_batch_page_init(void *page,
							   size_t page_size,
							   const TqBatchPageParams *params,
							   char *errmsg,
							   size_t errmsg_len);
extern bool tq_batch_page_set_next_block(void *page,
										 size_t page_size,
										 uint32_t next_block,
										 char *errmsg,
										 size_t errmsg_len);
extern bool tq_batch_page_has_capacity(const void *page,
									   size_t page_size,
									   bool *has_capacity,
									   char *errmsg,
									   size_t errmsg_len);
extern bool tq_batch_page_should_reclaim(const void *page,
										 size_t page_size,
										 bool *should_reclaim,
										 char *errmsg,
										 size_t errmsg_len);
extern bool tq_batch_page_read_header(const void *page,
									  size_t page_size,
									  TqBatchPageHeaderView *header,
									  char *errmsg,
									  size_t errmsg_len);
extern bool tq_batch_page_set_summary(void *page,
									  size_t page_size,
									  const TqBatchPageSummary *summary,
									  char *errmsg,
									  size_t errmsg_len);
extern bool tq_batch_page_get_summary(const void *page,
									  size_t page_size,
									  TqBatchPageSummary *summary,
									  char *errmsg,
									  size_t errmsg_len);
extern bool tq_batch_page_append_lane(void *page,
									  size_t page_size,
									  const TqTid *tid,
									  uint16_t *lane_index,
									  char *errmsg,
									  size_t errmsg_len);
extern bool tq_batch_page_get_tid(const void *page,
								  size_t page_size,
								  uint16_t lane_index,
								  TqTid *tid,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_batch_page_set_code(void *page,
								   size_t page_size,
								   uint16_t lane_index,
								   const uint8_t *code,
								   size_t code_len,
								   char *errmsg,
								   size_t errmsg_len);
extern bool tq_batch_page_get_code(const void *page,
								   size_t page_size,
								   uint16_t lane_index,
								   uint8_t *code,
								   size_t code_len,
								   char *errmsg,
								   size_t errmsg_len);
extern bool tq_batch_page_code_view(const void *page,
									size_t page_size,
									uint16_t lane_index,
									const uint8_t **code,
									size_t *code_len,
									char *errmsg,
									size_t errmsg_len);
extern bool tq_batch_page_mark_dead(void *page,
									size_t page_size,
									uint16_t lane_index,
									char *errmsg,
									size_t errmsg_len);
extern bool tq_batch_page_compact(void *page,
								  size_t page_size,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_batch_page_is_live(const void *page,
								  size_t page_size,
								  uint16_t lane_index,
								  bool *is_live,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_batch_page_next_live_lane(const void *page,
										 size_t page_size,
										 int start_lane,
										 uint16_t *lane_index,
										 char *errmsg,
										 size_t errmsg_len);

extern bool tq_batch_page_is_soa(const void *page, size_t page_size);
extern bool tq_batch_page_get_nibble_ptr(const void *page, size_t page_size,
										 const uint8_t **nibbles, uint32_t *dimension,
										 uint16_t *lane_count, char *errmsg, size_t errmsg_len);
extern bool tq_batch_page_get_gamma_ptr(const void *page, size_t page_size,
										const float **gammas, char *errmsg, size_t errmsg_len);
extern bool tq_batch_page_set_nibble_and_gamma(void *page, size_t page_size,
											   uint16_t lane_index, const uint8_t *nibbles,
											   uint32_t dimension, float gamma,
											   char *errmsg, size_t errmsg_len);
extern size_t tq_batch_page_soa_required_bytes(uint16_t lane_count, uint32_t dimension,
											   uint32_t representative_code_bytes);
extern bool tq_batch_page_can_fit_soa(size_t page_size, uint16_t lane_count,
									  uint32_t dimension, uint32_t representative_code_bytes);
extern bool tq_batch_page_set_representative_code(void *page, size_t page_size,
												  const uint8_t *code, size_t code_len,
												  char *errmsg, size_t errmsg_len);
extern bool tq_batch_page_get_representative_code_view(const void *page, size_t page_size,
													   const uint8_t **code, size_t *code_len,
													   char *errmsg, size_t errmsg_len);

#endif
