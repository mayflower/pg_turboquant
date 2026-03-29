#include "src/tq_scan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/tq_simd_avx2.h"

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static void
tq_candidate_swap(TqCandidateEntry *left, TqCandidateEntry *right)
{
	TqCandidateEntry tmp = *left;

	*left = *right;
	*right = tmp;
}

static int
tq_candidate_compare_worst(const TqCandidateEntry *left, const TqCandidateEntry *right)
{
	if (left->score > right->score)
		return 1;
	if (left->score < right->score)
		return -1;
	if (left->tid.block_number > right->tid.block_number)
		return 1;
	if (left->tid.block_number < right->tid.block_number)
		return -1;
	if (left->tid.offset_number > right->tid.offset_number)
		return 1;
	if (left->tid.offset_number < right->tid.offset_number)
		return -1;
	return 0;
}

static float
tq_dot_product_scalar(const float *left, const float *right, size_t len)
{
	float		sum = 0.0f;
	size_t		i = 0;

	for (i = 0; i < len; i++)
		sum += left[i] * right[i];

	return sum;
}

static float
tq_norm_squared_scalar(const float *values, size_t len)
{
	float		sum = 0.0f;
	size_t		i = 0;

	for (i = 0; i < len; i++)
		sum += values[i] * values[i];

	return sum;
}

static bool
tq_metric_distance_from_decoded_vector(TqDistanceKind distance,
									   const float *query_values,
									   size_t query_len,
									   const float *decoded_values,
									   size_t decoded_len,
									   float query_norm_squared,
									   float *distance_value,
									   char *errmsg,
									   size_t errmsg_len)
{
	float		dot_product = 0.0f;

	if (query_values == NULL || decoded_values == NULL || distance_value == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: decoded metric scorer requires query, decoded vector, and output");
		return false;
	}

	if (query_len != decoded_len)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: decoded metric scorer requires matching vector dimensions");
		return false;
	}

	dot_product = tq_dot_product_scalar(query_values, decoded_values, query_len);

	switch (distance)
	{
		case TQ_DISTANCE_COSINE:
			{
				float	decoded_norm_squared = tq_norm_squared_scalar(decoded_values, decoded_len);
				float	denominator = sqrtf(query_norm_squared * decoded_norm_squared);
				float	cosine_similarity = 0.0f;

				if (query_norm_squared <= 0.0f || decoded_norm_squared <= 0.0f)
				{
					*distance_value = 1.0f;
					return true;
				}

				cosine_similarity = dot_product / denominator;
				if (cosine_similarity > 1.0f)
					cosine_similarity = 1.0f;
				else if (cosine_similarity < -1.0f)
					cosine_similarity = -1.0f;

				*distance_value = 1.0f - cosine_similarity;
				return true;
			}
		case TQ_DISTANCE_IP:
			*distance_value = -dot_product;
			return true;
		case TQ_DISTANCE_L2:
			{
				float	decoded_norm_squared = tq_norm_squared_scalar(decoded_values, decoded_len);

				*distance_value = query_norm_squared + decoded_norm_squared - (2.0f * dot_product);
				if (*distance_value < 0.0f && *distance_value > -1e-6f)
					*distance_value = 0.0f;
				return true;
			}
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant scan: unsupported distance kind");
			return false;
	}
}

static void
tq_candidate_sift_up(TqCandidateHeap *heap, size_t index)
{
	while (index > 0)
	{
		size_t		parent = (index - 1) / 2;

		if (tq_candidate_compare_worst(&heap->entries[index], &heap->entries[parent]) <= 0)
			break;

		tq_candidate_swap(&heap->entries[index], &heap->entries[parent]);
		index = parent;
	}
}

static void
tq_candidate_sift_down(TqCandidateHeap *heap, size_t index)
{
	for (;;)
	{
		size_t		left = (index * 2) + 1;
		size_t		right = left + 1;
		size_t		worst = index;

		if (left < heap->count
			&& tq_candidate_compare_worst(&heap->entries[left], &heap->entries[worst]) > 0)
			worst = left;

		if (right < heap->count
			&& tq_candidate_compare_worst(&heap->entries[right], &heap->entries[worst]) > 0)
			worst = right;

		if (worst == index)
			break;

		tq_candidate_swap(&heap->entries[index], &heap->entries[worst]);
		index = worst;
	}
}

static int
tq_candidate_compare_best_qsort(const void *left, const void *right)
{
	const TqCandidateEntry *lhs = (const TqCandidateEntry *) left;
	const TqCandidateEntry *rhs = (const TqCandidateEntry *) right;

	if (lhs->score < rhs->score)
		return -1;
	if (lhs->score > rhs->score)
		return 1;
	if (lhs->tid.block_number < rhs->tid.block_number)
		return -1;
	if (lhs->tid.block_number > rhs->tid.block_number)
		return 1;
	if (lhs->tid.offset_number < rhs->tid.offset_number)
		return -1;
	if (lhs->tid.offset_number > rhs->tid.offset_number)
		return 1;
	return 0;
}

bool
tq_candidate_heap_init(TqCandidateHeap *heap, size_t capacity)
{
	if (heap == NULL || capacity == 0)
		return false;

	memset(heap, 0, sizeof(*heap));
	heap->entries = (TqCandidateEntry *) calloc(capacity, sizeof(TqCandidateEntry));
	if (heap->entries == NULL)
		return false;

	heap->capacity = capacity;
	return true;
}

void
tq_candidate_heap_reset(TqCandidateHeap *heap)
{
	if (heap == NULL)
		return;

	free(heap->entries);
	memset(heap, 0, sizeof(*heap));
}

bool
tq_candidate_heap_push(TqCandidateHeap *heap,
					   float score,
					   uint32_t block_number,
					   uint16_t offset_number)
{
	TqCandidateEntry entry;

	if (heap == NULL || heap->entries == NULL || heap->capacity == 0)
		return false;

	entry.score = score;
	entry.tid.block_number = block_number;
	entry.tid.offset_number = offset_number;

	if (heap->count < heap->capacity)
	{
		heap->entries[heap->count] = entry;
		tq_candidate_sift_up(heap, heap->count);
		heap->count++;
		heap->sorted = false;
		return true;
	}

	if (tq_candidate_compare_worst(&entry, &heap->entries[0]) >= 0)
		return true;

	heap->entries[0] = entry;
	tq_candidate_sift_down(heap, 0);
	heap->sorted = false;
	return true;
}

bool
tq_candidate_heap_pop_best(TqCandidateHeap *heap, TqCandidateEntry *entry)
{
	if (heap == NULL || entry == NULL || heap->entries == NULL)
		return false;

	if (!heap->sorted)
	{
		qsort(heap->entries, heap->count, sizeof(TqCandidateEntry),
			  tq_candidate_compare_best_qsort);
		heap->sorted = true;
		heap->pop_index = 0;
	}

	if (heap->pop_index >= heap->count)
		return false;

	*entry = heap->entries[heap->pop_index++];
	return true;
}

bool
tq_metric_distance_from_ip_score(TqDistanceKind distance,
								 float ip_score,
								 float *distance_value,
								 char *errmsg,
								 size_t errmsg_len)
{
	if (distance_value == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: distance output must be non-null");
		return false;
	}

	switch (distance)
	{
		case TQ_DISTANCE_COSINE:
			*distance_value = 1.0f - ip_score;
			return true;
		case TQ_DISTANCE_IP:
			*distance_value = -ip_score;
			return true;
		case TQ_DISTANCE_L2:
			*distance_value = 2.0f - (2.0f * ip_score);
			if (*distance_value < 0.0f && *distance_value > -1e-6f)
				*distance_value = 0.0f;
			return true;
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant scan: unsupported distance kind");
			return false;
	}
}

bool
tq_batch_page_scan_prod(const void *page,
						size_t page_size,
						const TqProdCodecConfig *config,
						TqDistanceKind distance,
						const TqProdLut *lut,
						const float *query_values,
						size_t query_len,
						TqCandidateHeap *heap,
						char *errmsg,
						size_t errmsg_len)
{
	TqBatchPageHeaderView header;
	uint8_t    *code = NULL;
	float	   *decoded = NULL;
	float		query_norm_squared = 0.0f;
	uint16_t	lane = 0;

	if (page == NULL || config == NULL || lut == NULL || heap == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: page, codec, lut, and heap must be non-null");
		return false;
	}

	memset(&header, 0, sizeof(header));

	if (query_values == NULL || query_len != config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: query values must match codec dimension");
		return false;
	}

	if (!tq_batch_page_read_header(page, page_size, &header, errmsg, errmsg_len))
		return false;

	code = (uint8_t *) malloc(header.code_bytes);
	if (code == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: out of memory");
		return false;
	}

	decoded = (float *) malloc(sizeof(float) * query_len);
	if (decoded == NULL)
	{
		free(code);
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: out of memory");
		return false;
	}

	query_norm_squared = tq_norm_squared_scalar(query_values, query_len);

	if (tq_batch_page_next_live_lane(page, page_size, -1, &lane, errmsg, errmsg_len))
	{
		do
		{
			TqTid		tid;
			float		distance_value = 0.0f;

			memset(&tid, 0, sizeof(tid));
			if (!tq_batch_page_get_tid(page, page_size, lane, &tid, errmsg, errmsg_len)
				|| !tq_batch_page_get_code(page, page_size, lane, code, header.code_bytes, errmsg, errmsg_len))
			{
				free(decoded);
				free(code);
				return false;
			}

			if (!tq_prod_decode(config, code, header.code_bytes, decoded, query_len,
								errmsg, errmsg_len))
			{
				free(decoded);
				free(code);
				return false;
			}

			if (!tq_metric_distance_from_decoded_vector(distance,
														query_values,
														query_len,
														decoded,
														query_len,
														query_norm_squared,
														&distance_value,
														errmsg,
														errmsg_len))
			{
				free(decoded);
				free(code);
				return false;
			}

			if (!tq_candidate_heap_push(heap, distance_value, tid.block_number, tid.offset_number))
			{
				free(decoded);
				free(code);
				return false;
			}
		} while (tq_batch_page_next_live_lane(page, page_size, (int) lane, &lane, errmsg, errmsg_len));
	}

	free(decoded);
	free(code);
	return true;
}

bool
tq_batch_page_scan_prod_cosine(const void *page,
							   size_t page_size,
							   const TqProdCodecConfig *config,
							   const TqProdLut *lut,
							   const float *query_values,
							   size_t query_len,
							   TqCandidateHeap *heap,
							   char *errmsg,
							   size_t errmsg_len)
{
	return tq_batch_page_scan_prod(page, page_size, config, TQ_DISTANCE_COSINE,
								   lut, query_values, query_len, heap,
								   errmsg, errmsg_len);
}
