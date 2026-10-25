#define VIZTOKEN_BUILD_DLL 1
#define TTKN_BUILD_DLL 1
#include "viztoken.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#define PCRE2_STATIC 1
#include <pcre2.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define TTKN_HAS_SSE2 1
#if defined(__AVX2__)
#define TTKN_HAS_AVX2 1
#endif
#endif

#define TTKN_MAGIC_V3 "TIKTALKIN_SWISS3"
#define TTKN_MAGIC_V4 "TIKTALKIN_SWISS4"
#define TTKN_MAGIC_V5 "VIZTOK_SWISS5___"
#define TTKN_MAGIC_LEN 16
#define VIZTOK_FAST_MAGIC 0x565A5435U

#define DEFAULT_BASE_VOCAB_SIZE 151643U
#define DEFAULT_TOTAL_VOCAB_SIZE 184704U
#define DEFAULT_EOT_TOKEN_ID 151643U
#define INF_RANK 0xFFFFFFFFU
#define SLOT_NONE 0xFFFFFFFFU
#define SLOT_SPECIAL_BIT 0x80000000U
#define SSO_MAX_LEN 16U
#define CTRL_EMPTY 0x80
#define CTRL_SENTINEL 0xFE
#define FLAT_MERGE_MAX_LEN 64U
#define BPE_CACHE_SLOTS 512U
#define BPE_CACHE_MAX_TOKENS 32U

#pragma pack(push, 1)
typedef struct {
    uint8_t  magic[TTKN_MAGIC_LEN];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t max_token_len;
    uint32_t table_capacity;
    uint64_t ctrl_offset;
    uint64_t slots_offset;
    uint64_t string_data_offset;
    uint64_t string_data_bytes;
    uint32_t sso_token_count;
    uint8_t  reserved[60];
} ttkn_header_v3_t;

typedef struct {
    uint8_t  magic[TTKN_MAGIC_LEN];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t total_vocab_size;
    uint32_t max_token_len;
    uint32_t table_capacity;
    uint32_t sso_token_count;
    uint64_t ctrl_offset;
    uint64_t slots_offset;
    uint64_t string_data_offset;
    uint64_t string_data_bytes;
    uint64_t regex_offset;
    uint32_t regex_bytes;
    uint32_t specials_count;
    uint64_t specials_offset;
    uint32_t eot_token_id;
    uint32_t fast_tables_magic;
    uint64_t byte_ranks_offset;
    uint64_t pair_ranks_offset;
    uint64_t id_to_slot_offset;
} ttkn_header_v4_t;

typedef struct {
    uint64_t hash;
    uint32_t rank;
    uint16_t length;
    uint8_t  is_inline;
    uint8_t  reserved;
    union {
        uint8_t  inline_bytes[SSO_MAX_LEN];
        struct {
            uint64_t offset;
            uint8_t  pad[8];
        } heap;
    } storage;
} ttkn_slot_t;
#pragma pack(pop)

typedef char viztok_hdr_assert[(sizeof(ttkn_header_v4_t) == 128) ? 1 : -1];

typedef struct {
    void *raw_handle;
    void *mapping_handle;
    uint8_t *base_addr;
    uint64_t total_bytes;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t total_vocab_size;
    uint32_t table_capacity;
    uint32_t max_token_len;
    uint64_t string_data_bytes;
    uint32_t sso_token_count;
    const uint8_t *ctrl;
    const ttkn_slot_t *slots;
    const uint8_t *string_data;
    const char *regex_pattern;
    uint32_t regex_bytes;
    const viztoken_special_disk_t *specials;
    uint32_t specials_count;
    uint32_t eot_token_id;
    const uint32_t *mapped_byte_ranks;
    const uint32_t *mapped_pair_ranks;
    const uint32_t *mapped_id_to_slot;
} ttkn_mmap_t;

typedef struct {
    pcre2_code *code;
    uint64_t uid;
} ttkn_regex_t;

typedef struct {
    char str[64];
    uint32_t len;
    int32_t id;
} special_entry_t;

typedef struct {
    uint32_t offset;
    uint32_t rank;
    uint32_t token_id;
} ttkn_part_t;

typedef struct {
    uint64_t hash;
    uint64_t ctx_uid;
    uint16_t byte_len;
    uint16_t token_count;
    uint8_t  bytes[FLAT_MERGE_MAX_LEN];
    int32_t  tokens[BPE_CACHE_MAX_TOKENS];
} bpe_cache_slot_t;

#if defined(_MSC_VER)
static __declspec(thread) bpe_cache_slot_t tls_bpe_cache[BPE_CACHE_SLOTS];
#else
static __thread bpe_cache_slot_t tls_bpe_cache[BPE_CACHE_SLOTS];
#endif

struct viztoken_ctx_t {
    ttkn_mmap_t vm;
    ttkn_regex_t *regex;
    int is_standard_qwen_regex;
    int all_specials_start_with_lt;
    char **id_to_str_fallback;
    uint32_t *id_to_len_fallback;
    special_entry_t *specials;
    uint32_t special_count;
    uint8_t special_prefix_mask[256];
    uint32_t max_allocated_id;
    uint32_t base_vocab_size;
    uint32_t total_vocab_size;
    uint32_t eot_token_id;
    const uint32_t *byte_ranks;
    const uint32_t *pair_ranks_2byte;
    const uint32_t *id_to_slot;
    uint32_t *owned_byte_ranks;
    uint32_t *owned_pair_ranks;
};

static const char *DEFAULT_QWEN_PATTERN =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

static const char *DEFAULT_QWEN35_PATTERN =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

#if defined(_WIN32)
static int utf8_to_wide(const char *utf8_path, wchar_t *out_wpath, size_t max_wchars) {
    if (!utf8_path || !out_wpath || max_wchars == 0) return -1;
    int n = MultiByteToWideChar(CP_UTF8, 0, utf8_path, -1, out_wpath, (int)max_wchars);
    if (n <= 0) {
        n = MultiByteToWideChar(CP_ACP, 0, utf8_path, -1, out_wpath, (int)max_wchars);
    }
    return (n > 0) ? 0 : -1;
}

static FILE *open_file_utf8(const char *path, const wchar_t *wmode) {
    wchar_t wpath[2048];
    if (utf8_to_wide(path, wpath, 2048) != 0) return NULL;
    return _wfopen(wpath, wmode);
}

static void atomic_replace_file_utf8(const char *tmp_path, const char *dst_path) {
    wchar_t wtmp[2048];
    wchar_t wdst[2048];
    if (utf8_to_wide(tmp_path, wtmp, 2048) == 0 && utf8_to_wide(dst_path, wdst, 2048) == 0) {
        SetFileAttributesW(wdst, FILE_ATTRIBUTE_NORMAL);
        MoveFileExW(wtmp, wdst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    }
}
#else
static FILE *open_file_utf8(const char *path, const char *mode) {
    return fopen(path, mode);
}

static void atomic_replace_file_utf8(const char *tmp_path, const char *dst_path) {
    chmod(dst_path, 0666);
    rename(tmp_path, dst_path);
}
#endif

static inline uint32_t ttkn_ctz32(uint32_t mask) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_ctz(mask);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward(&idx, mask);
    return (uint32_t)idx;
#else
    uint32_t idx = 0;
    while ((mask & 1) == 0) {
        mask >>= 1;
        idx++;
    }
    return idx;
#endif
}

static inline bool ttkn_memeq(const uint8_t *a, const uint8_t *b, uint32_t len) {
    if (len <= 4) {
        if (len == 1) return a[0] == b[0];
        if (len == 2) {
            uint16_t u, v;
            memcpy(&u, a, 2);
            memcpy(&v, b, 2);
            return u == v;
        }
        if (len == 3) {
            uint16_t u, v;
            memcpy(&u, a, 2);
            memcpy(&v, b, 2);
            return (u == v) && (a[2] == b[2]);
        }
        if (len == 4) {
            uint32_t u, v;
            memcpy(&u, a, 4);
            memcpy(&v, b, 4);
            return u == v;
        }
        return true;
    }
    if (len <= 8) {
        uint32_t u1, v1, u2, v2;
        memcpy(&u1, a, 4);
        memcpy(&v1, b, 4);
        memcpy(&u2, a + len - 4, 4);
        memcpy(&v2, b + len - 4, 4);
        return (u1 == v1) && (u2 == v2);
    }
    if (len <= 16) {
        uint64_t u1, v1, u2, v2;
        memcpy(&u1, a, 8);
        memcpy(&v1, b, 8);
        memcpy(&u2, a + len - 8, 8);
        memcpy(&v2, b + len - 8, 8);
        return (u1 == v1) && (u2 == v2);
    }
    return memcmp(a, b, len) == 0;
}

static inline uint64_t ttkn_hash64(const uint8_t *data, uint32_t len) {
    const uint64_t prime1 = 0x9E3779B185EBCA87ULL;
    const uint64_t prime2 = 0xC2B2AE3D27D4EB4FULL;
    const uint64_t prime3 = 0x165667B19E3779F9ULL;
    const uint64_t prime4 = 0x85EBCA77C2B2AE63ULL;
    const uint64_t prime5 = 0x27D4EB2F165667C5ULL;

    uint64_t h = prime5 + ((uint64_t)len * prime1);
    const uint8_t *p = data;
    const uint8_t *end = data + len;

    while (p + 8 <= end) {
        uint64_t k;
        memcpy(&k, p, 8);
        k *= prime2;
        k = (k << 31) | (k >> 33);
        k *= prime1;
        h ^= k;
        h = ((h << 27) | (h >> 37)) * prime1 + prime4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t k;
        memcpy(&k, p, 4);
        h ^= (uint64_t)k * prime1;
        h = ((h << 23) | (h >> 41)) * prime2 + prime3;
        p += 4;
    }
    while (p < end) {
        h ^= ((uint64_t)(*p)) * prime5;
        h = ((h << 11) | (h >> 53)) * prime1;
        p++;
    }
    h ^= h >> 33;
    h *= prime2;
    h ^= h >> 29;
    h *= prime3;
    h ^= h >> 32;
    return h ? h : 0x8000000000000001ULL;
}

static void ttkn_mmap_close(ttkn_mmap_t *vm) {
    if (!vm) return;
#if defined(_WIN32)
    if (vm->base_addr) UnmapViewOfFile(vm->base_addr);
    if (vm->mapping_handle) CloseHandle((HANDLE)vm->mapping_handle);
    if (vm->raw_handle) CloseHandle((HANDLE)vm->raw_handle);
#else
    if (vm->base_addr) munmap(vm->base_addr, (size_t)vm->total_bytes);
    if ((intptr_t)vm->raw_handle > 0) close((int)(intptr_t)vm->raw_handle);
#endif
    memset(vm, 0, sizeof(ttkn_mmap_t));
}

static int ttkn_mmap_open(ttkn_mmap_t *vm, const char *path) {
    if (!vm || !path) return -1;
    memset(vm, 0, sizeof(ttkn_mmap_t));
#if defined(_WIN32)
    wchar_t wpath[2048];
    if (utf8_to_wide(path, wpath, 2048) != 0) return -2;
    HANDLE f = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) return -2;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz)) {
        CloseHandle(f);
        return -3;
    }
    vm->total_bytes = (uint64_t)sz.QuadPart;
    HANDLE m = CreateFileMappingW(f, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m) {
        CloseHandle(f);
        return -4;
    }
    uint8_t *base = (uint8_t *)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        CloseHandle(m);
        CloseHandle(f);
        return -5;
    }
    vm->raw_handle = (void *)f;
    vm->mapping_handle = (void *)m;
    vm->base_addr = base;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -2;
    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        close(fd);
        return -3;
    }
    vm->total_bytes = (uint64_t)sb.st_size;
    uint8_t *base = (uint8_t *)mmap(NULL, (size_t)vm->total_bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        return -5;
    }
    vm->raw_handle = (void *)(intptr_t)fd;
    vm->base_addr = base;
#endif
    if (vm->total_bytes < TTKN_MAGIC_LEN) {
        ttkn_mmap_close(vm);
        return -6;
    }
    if (memcmp(vm->base_addr, TTKN_MAGIC_V4, TTKN_MAGIC_LEN) == 0 ||
        memcmp(vm->base_addr, TTKN_MAGIC_V5, TTKN_MAGIC_LEN) == 0) {
        if (vm->total_bytes < sizeof(ttkn_header_v4_t)) {
            ttkn_mmap_close(vm);
            return -7;
        }
        const ttkn_header_v4_t *hdr = (const ttkn_header_v4_t *)vm->base_addr;
        vm->version = hdr->version;
        vm->vocab_size = hdr->vocab_size;
        vm->total_vocab_size = hdr->total_vocab_size;
        vm->max_token_len = hdr->max_token_len;
        vm->table_capacity = hdr->table_capacity;
        vm->sso_token_count = hdr->sso_token_count;
        vm->string_data_bytes = hdr->string_data_bytes;
        vm->ctrl = (const uint8_t *)(vm->base_addr + hdr->ctrl_offset);
        vm->slots = (const ttkn_slot_t *)(vm->base_addr + hdr->slots_offset);
        vm->string_data = (const uint8_t *)(vm->base_addr + hdr->string_data_offset);
        if (hdr->regex_offset > 0 && hdr->regex_bytes > 0) {
            vm->regex_pattern = (const char *)(vm->base_addr + hdr->regex_offset);
            vm->regex_bytes = hdr->regex_bytes;
        }
        if (hdr->specials_offset > 0 && hdr->specials_count > 0) {
            vm->specials = (const viztoken_special_disk_t *)(vm->base_addr + hdr->specials_offset);
            vm->specials_count = hdr->specials_count;
        }
        vm->eot_token_id = hdr->eot_token_id;
        if (hdr->fast_tables_magic == VIZTOK_FAST_MAGIC &&
            hdr->byte_ranks_offset > 0 &&
            hdr->pair_ranks_offset > 0 &&
            hdr->id_to_slot_offset > 0 &&
            hdr->id_to_slot_offset + (uint64_t)hdr->total_vocab_size * sizeof(uint32_t) <= vm->total_bytes) {
            vm->mapped_byte_ranks = (const uint32_t *)(vm->base_addr + hdr->byte_ranks_offset);
            vm->mapped_pair_ranks = (const uint32_t *)(vm->base_addr + hdr->pair_ranks_offset);
            vm->mapped_id_to_slot = (const uint32_t *)(vm->base_addr + hdr->id_to_slot_offset);
        }
        return 0;
    }
    if (memcmp(vm->base_addr, TTKN_MAGIC_V3, TTKN_MAGIC_LEN) == 0) {
        if (vm->total_bytes < sizeof(ttkn_header_v3_t)) {
            ttkn_mmap_close(vm);
            return -8;
        }
        const ttkn_header_v3_t *hdr = (const ttkn_header_v3_t *)vm->base_addr;
        vm->version = hdr->version;
        vm->vocab_size = hdr->vocab_size;
        vm->total_vocab_size = DEFAULT_TOTAL_VOCAB_SIZE;
        vm->max_token_len = hdr->max_token_len;
        vm->table_capacity = hdr->table_capacity;
        vm->sso_token_count = hdr->sso_token_count;
        vm->string_data_bytes = hdr->string_data_bytes;
        vm->ctrl = (const uint8_t *)(vm->base_addr + hdr->ctrl_offset);
        vm->slots = (const ttkn_slot_t *)(vm->base_addr + hdr->slots_offset);
        vm->string_data = (const uint8_t *)(vm->base_addr + hdr->string_data_offset);
        vm->regex_pattern = DEFAULT_QWEN_PATTERN;
        vm->regex_bytes = (uint32_t)strlen(DEFAULT_QWEN_PATTERN) + 1;
        vm->specials = NULL;
        vm->specials_count = 0;
        vm->eot_token_id = DEFAULT_EOT_TOKEN_ID;
        return 0;
    }
    ttkn_mmap_close(vm);
    return -9;
}

static inline int ttkn_swiss_lookup_with_hash(
    const ttkn_mmap_t *vm,
    const uint8_t *data,
    uint32_t len,
    uint64_t h,
    uint32_t *out_rank
) {
    uint32_t group_count = vm->table_capacity / 16;
    uint32_t group_mask = group_count - 1;
    uint32_t group = (uint32_t)((h >> 7) & group_mask);
    uint8_t h2 = (uint8_t)(h & 0x7F);

    const uint8_t *ctrl = vm->ctrl;
    const ttkn_slot_t *slots = vm->slots;
    const uint8_t *str_data = vm->string_data;

#if defined(TTKN_HAS_SSE2)
    __m128i match_vec = _mm_set1_epi8((char)h2);
    __m128i empty_vec = _mm_set1_epi8((char)CTRL_EMPTY);
#endif
    uint32_t step = 0;
    while (1) {
        uint32_t group_offset = group * 16;
#if defined(TTKN_HAS_SSE2)
        __m128i ctrl_vec = _mm_loadu_si128((const __m128i *)(ctrl + group_offset));
        __m128i eq = _mm_cmpeq_epi8(ctrl_vec, match_vec);
        uint32_t match_mask = (uint32_t)_mm_movemask_epi8(eq) & 0xFFFFU;
#else
        uint32_t match_mask = 0;
        for (uint32_t i = 0; i < 16; i++) {
            if (ctrl[group_offset + i] == h2) match_mask |= (1U << i);
        }
#endif
        while (match_mask != 0) {
            uint32_t bit = ttkn_ctz32(match_mask);
            const ttkn_slot_t *slot = &slots[group_offset + bit];
            if (slot->hash == h && slot->length == len) {
                const uint8_t *cand = slot->is_inline ? slot->storage.inline_bytes : (str_data + (size_t)slot->storage.heap.offset);
                if (ttkn_memeq(cand, data, len)) {
                    *out_rank = slot->rank;
                    return 1;
                }
            }
            match_mask &= (match_mask - 1);
        }
#if defined(TTKN_HAS_SSE2)
        __m128i is_empty = _mm_cmpeq_epi8(ctrl_vec, empty_vec);
        uint32_t empty_mask = (uint32_t)_mm_movemask_epi8(is_empty) & 0xFFFFU;
#else
        uint32_t empty_mask = 0;
        for (uint32_t i = 0; i < 16; i++) {
            if (ctrl[group_offset + i] == CTRL_EMPTY) empty_mask |= (1U << i);
        }
#endif
        if (empty_mask != 0) {
            return 0;
        }
        step++;
        if (step >= group_count) {
            return 0;
        }
        group = (group + step) & group_mask;
    }
}

static inline int ttkn_swiss_lookup(const ttkn_mmap_t *vm, const uint8_t *data, uint32_t len, uint32_t *out_rank) {
    if (!vm || !data || len == 0 || !vm->ctrl || !vm->slots) return 0;
    uint64_t h = ttkn_hash64(data, len);
    return ttkn_swiss_lookup_with_hash(vm, data, len, h, out_rank);
}

static inline uint32_t ttkn_query_pair_rank(
    const viztoken_ctx_t *ctx,
    const uint8_t *piece,
    uint32_t start,
    uint32_t len
) {
    if (len == 2) {
        uint16_t key;
        memcpy(&key, piece + start, 2);
        return ctx->pair_ranks_2byte[key];
    }
    uint32_t r = INF_RANK;
    ttkn_swiss_lookup(&ctx->vm, piece + start, len, &r);
    return r;
}

static const int8_t b64_table[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
};

static uint32_t b64_decode(const char *in, uint32_t in_len, uint8_t *out) {
    uint32_t out_len = 0;
    uint32_t buf = 0;
    int bits = 0;
    for (uint32_t i = 0; i < in_len; i++) {
        uint8_t c = (uint8_t)in[i];
        if (c == '=') break;
        int8_t d = b64_table[c];
        if (d < 0) continue;
        buf = (buf << 6) | (uint32_t)d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[out_len++] = (uint8_t)((buf >> bits) & 0xFF);
            buf &= (1U << bits) - 1;
        }
    }
    return out_len;
}

static uint32_t next_pow2(uint32_t x) {
    if (x <= 1) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

static int flush_and_commit_file(FILE *f) {
    if (!f) return -1;
    if (fflush(f) != 0) return -2;
#if defined(_WIN32)
    int fd = _fileno(f);
    if (fd >= 0) {
        HANDLE h = (HANDLE)_get_osfhandle(fd);
        if (h != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(h);
        }
    }
#else
    int fd = fileno(f);
    if (fd >= 0) {
        fsync(fd);
    }
#endif
    return 0;
}

static uint32_t populate_default_qwen_specials(viztoken_special_disk_t *out_specials) {
    uint32_t n = 0;
    #define EMIT_SPEC(text, tid) do { \
        if (out_specials) { \
            out_specials[n].id = (int32_t)(tid); \
            strncpy(out_specials[n].literal, (text), 57); \
            out_specials[n].literal[57] = '\0'; \
            out_specials[n].length = (uint16_t)strlen(out_specials[n].literal); \
        } \
        n++; \
    } while (0)

    EMIT_SPEC("<|endoftext|>", 151643);
    EMIT_SPEC("<|im_start|>",   151644);
    EMIT_SPEC("<|im_end|>",     151645);
    EMIT_SPEC("<R>",            151646);
    EMIT_SPEC("<S>",            151647);
    EMIT_SPEC("<X>",            151648);
    EMIT_SPEC("<mask >",        151649);
    EMIT_SPEC("<sep>",          151650);
    for (int i = 0; i < 196; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "<extra_%d>", i);
        EMIT_SPEC(buf, 151651 + i);
    }
    EMIT_SPEC("<abc>",          151847);
    EMIT_SPEC("</abc>",         151848);
    EMIT_SPEC("<extra_198>",    151849);
    EMIT_SPEC("<extra_199>",    151850);
    EMIT_SPEC("<music_start>",  151851);
    EMIT_SPEC("<music_end>",    151852);
    EMIT_SPEC("<latent_start>", 184621);
    EMIT_SPEC("<latent_end>",   184622);
    EMIT_SPEC("<latent_pad>",   184623);
    #undef EMIT_SPEC
    return n;
}

VIZTOKEN_API int32_t viztoken_compile_vocab_v5(
    const char *in_tiktoken_path,
    const char *out_bin_path,
    const char *regex_pattern,
    const viztoken_special_disk_t *specials,
    uint32_t specials_count,
    uint32_t eot_token_id,
    const char *out_telemetry_json,
    viztoken_telemetry_t *out_telemetry
) {
    clock_t t0 = clock();
#if defined(_WIN32)
    FILE *f_in = open_file_utf8(in_tiktoken_path, L"rb");
#else
    FILE *f_in = open_file_utf8(in_tiktoken_path, "rb");
#endif
    if (!f_in) return -1;

    fseek(f_in, 0, SEEK_END);
    long f_size = ftell(f_in);
    fseek(f_in, 0, SEEK_SET);

    char *text_buf = (char *)malloc((size_t)f_size + 1);
    if (!text_buf) {
        fclose(f_in);
        return -2;
    }
    if (fread(text_buf, 1, (size_t)f_size, f_in) != (size_t)f_size) {
        free(text_buf);
        fclose(f_in);
        return -3;
    }
    text_buf[f_size] = '\0';
    fclose(f_in);

    uint32_t capacity_records = 262144;
    uint8_t **tokens = (uint8_t **)malloc(sizeof(uint8_t *) * capacity_records);
    uint32_t *lengths = (uint32_t *)malloc(sizeof(uint32_t) * capacity_records);
    uint32_t *ranks = (uint32_t *)malloc(sizeof(uint32_t) * capacity_records);

    uint32_t count = 0;
    uint32_t max_len = 0;
    uint64_t total_heap_str_bytes = 0;
    uint32_t sso_count = 0;

    char *cursor = text_buf;
    uint8_t decode_scratch[4096];

    while (*cursor) {
        while (*cursor == '\r' || *cursor == '\n' || *cursor == ' ') cursor++;
        if (!*cursor) break;

        char *token_b64 = cursor;
        while (*cursor && *cursor != ' ' && *cursor != '\t' && *cursor != '\r' && *cursor != '\n') {
            cursor++;
        }
        uint32_t b64_len = (uint32_t)(cursor - token_b64);
        while (*cursor == ' ' || *cursor == '\t') cursor++;

        char *rank_str = cursor;
        while (*cursor && *cursor != '\r' && *cursor != '\n') {
            cursor++;
        }
        uint32_t rank = (uint32_t)strtoul(rank_str, NULL, 10);
        uint32_t dec_len = b64_decode(token_b64, b64_len, decode_scratch);

        uint8_t *allocated = (uint8_t *)malloc(dec_len);
        memcpy(allocated, decode_scratch, dec_len);

        if (count >= capacity_records) {
            capacity_records *= 2;
            tokens = (uint8_t **)realloc(tokens, sizeof(uint8_t *) * capacity_records);
            lengths = (uint32_t *)realloc(lengths, sizeof(uint32_t) * capacity_records);
            ranks = (uint32_t *)realloc(ranks, sizeof(uint32_t) * capacity_records);
        }
        tokens[count] = allocated;
        lengths[count] = dec_len;
        ranks[count] = rank;
        if (dec_len <= SSO_MAX_LEN) {
            sso_count++;
        } else {
            total_heap_str_bytes += dec_len;
        }
        if (dec_len > max_len) max_len = dec_len;
        count++;
    }
    free(text_buf);

    uint32_t table_capacity = next_pow2(count * 2);
    if (table_capacity < 16) table_capacity = 16;
    uint32_t group_count = table_capacity / 16;
    uint32_t group_mask = group_count - 1;

    uint64_t ctrl_padded_bytes = ((uint64_t)table_capacity + 16ULL + 63ULL) & ~63ULL;
    uint8_t *ctrl = (uint8_t *)malloc((size_t)ctrl_padded_bytes);
    memset(ctrl, CTRL_EMPTY, (size_t)ctrl_padded_bytes);

    ttkn_slot_t *slots = (ttkn_slot_t *)calloc(table_capacity, sizeof(ttkn_slot_t));
    uint8_t *str_blob = total_heap_str_bytes > 0 ? (uint8_t *)malloc((size_t)total_heap_str_bytes) : NULL;
    uint32_t current_heap_offset = 0;

    uint32_t max_probe_groups = 0;
    uint64_t sum_probe_groups = 0;
    uint32_t *item_probes = (uint32_t *)malloc(sizeof(uint32_t) * count);

    uint32_t *pre_byte_ranks = (uint32_t *)malloc(256 * sizeof(uint32_t));
    uint32_t *pre_pair_ranks = (uint32_t *)malloc(65536 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 256; i++) pre_byte_ranks[i] = INF_RANK;
    for (uint32_t i = 0; i < 65536; i++) pre_pair_ranks[i] = INF_RANK;

    uint32_t max_seen_id = count;
    for (uint32_t i = 0; i < count; i++) {
        if (ranks[i] > max_seen_id) max_seen_id = ranks[i];
    }

    viztoken_special_disk_t *local_specials = NULL;
    uint32_t effective_specials_count = specials_count;
    if (!specials || specials_count == 0) {
        local_specials = (viztoken_special_disk_t *)malloc(sizeof(viztoken_special_disk_t) * 512);
        effective_specials_count = populate_default_qwen_specials(local_specials);
    } else {
        local_specials = (viztoken_special_disk_t *)malloc(sizeof(viztoken_special_disk_t) * specials_count);
        memcpy(local_specials, specials, sizeof(viztoken_special_disk_t) * specials_count);
    }

    for (uint32_t i = 0; i < effective_specials_count; i++) {
        if ((uint32_t)local_specials[i].id > max_seen_id) {
            max_seen_id = (uint32_t)local_specials[i].id;
        }
    }
    uint32_t total_vocab_calc = max_seen_id + 1;
    if (total_vocab_calc < DEFAULT_TOTAL_VOCAB_SIZE) {
        total_vocab_calc = DEFAULT_TOTAL_VOCAB_SIZE;
    }

    uint32_t *pre_id_to_slot = (uint32_t *)malloc((size_t)total_vocab_calc * sizeof(uint32_t));
    for (uint32_t i = 0; i < total_vocab_calc; i++) pre_id_to_slot[i] = SLOT_NONE;

    for (uint32_t i = 0; i < count; i++) {
        uint8_t *tok = tokens[i];
        uint32_t len = lengths[i];
        uint32_t r = ranks[i];

        if (len == 1) {
            pre_byte_ranks[tok[0]] = r;
        } else if (len == 2) {
            uint16_t key;
            memcpy(&key, tok, 2);
            pre_pair_ranks[key] = r;
        }

        uint64_t h = ttkn_hash64(tok, len);
        uint8_t h2 = (uint8_t)(h & 0x7F);
        uint32_t ideal_group = (uint32_t)((h >> 7) & group_mask);
        uint32_t group = ideal_group;
        uint32_t step = 0;
        uint32_t inserted_slot = 0xFFFFFFFFU;

        while (1) {
            uint32_t base_slot = group * 16;
            for (uint32_t slot_idx = 0; slot_idx < 16; slot_idx++) {
                if (ctrl[base_slot + slot_idx] == CTRL_EMPTY) {
                    inserted_slot = base_slot + slot_idx;
                    break;
                }
            }
            if (inserted_slot != 0xFFFFFFFFU) {
                break;
            }
            step++;
            group = (group + step) & group_mask;
        }

        ctrl[inserted_slot] = h2;
        slots[inserted_slot].hash = h;
        slots[inserted_slot].rank = r;
        slots[inserted_slot].length = (uint16_t)len;
        if (len <= SSO_MAX_LEN) {
            slots[inserted_slot].is_inline = 1;
            memcpy(slots[inserted_slot].storage.inline_bytes, tok, len);
        } else {
            slots[inserted_slot].is_inline = 0;
            memcpy(str_blob + current_heap_offset, tok, len);
            slots[inserted_slot].storage.heap.offset = current_heap_offset;
            current_heap_offset += len;
        }

        if (r < total_vocab_calc) {
            pre_id_to_slot[r] = inserted_slot;
        }

        uint32_t probe_dist = (group + group_count - ideal_group) & group_mask;
        item_probes[i] = probe_dist;
        if (probe_dist > max_probe_groups) max_probe_groups = probe_dist;
        sum_probe_groups += probe_dist;
        free(tok);
    }
    free(tokens);
    free(lengths);
    free(ranks);

    for (uint32_t i = 0; i < effective_specials_count; i++) {
        int32_t sid = local_specials[i].id;
        if (sid >= 0 && (uint32_t)sid < total_vocab_calc) {
            pre_id_to_slot[(uint32_t)sid] = SLOT_SPECIAL_BIT | i;
        }
    }

    memcpy(ctrl + table_capacity, ctrl, 16);

    double mean_probe = count > 0 ? ((double)sum_probe_groups / (double)count) : 0.0;
    double var_accum = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        double diff = (double)item_probes[i] - mean_probe;
        var_accum += diff * diff;
    }
    double probe_var = count > 0 ? (var_accum / (double)count) : 0.0;
    free(item_probes);

    const char *chosen_regex = (regex_pattern && regex_pattern[0] != '\0') ? regex_pattern : DEFAULT_QWEN_PATTERN;
    uint32_t regex_raw_len = (uint32_t)strlen(chosen_regex) + 1;

    uint64_t header_padded_bytes = (sizeof(ttkn_header_v4_t) + 63ULL) & ~63ULL;
    uint64_t slots_padded_bytes = ((uint64_t)table_capacity * sizeof(ttkn_slot_t) + 63ULL) & ~63ULL;
    uint64_t string_padded_bytes = (total_heap_str_bytes + 63ULL) & ~63ULL;
    uint64_t regex_padded_bytes = ((uint64_t)regex_raw_len + 63ULL) & ~63ULL;
    uint64_t specials_raw_bytes = (uint64_t)effective_specials_count * sizeof(viztoken_special_disk_t);
    uint64_t specials_padded_bytes = (specials_raw_bytes + 63ULL) & ~63ULL;

    uint64_t byte_ranks_raw_bytes = 256ULL * sizeof(uint32_t);
    uint64_t byte_ranks_padded_bytes = (byte_ranks_raw_bytes + 63ULL) & ~63ULL;
    uint64_t pair_ranks_raw_bytes = 65536ULL * sizeof(uint32_t);
    uint64_t pair_ranks_padded_bytes = (pair_ranks_raw_bytes + 63ULL) & ~63ULL;
    uint64_t id_to_slot_raw_bytes = (uint64_t)total_vocab_calc * sizeof(uint32_t);
    uint64_t id_to_slot_padded_bytes = (id_to_slot_raw_bytes + 63ULL) & ~63ULL;

    ttkn_header_v4_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, TTKN_MAGIC_V4, TTKN_MAGIC_LEN);
    hdr.version = 5;
    hdr.vocab_size = count;
    hdr.total_vocab_size = total_vocab_calc;
    hdr.max_token_len = max_len;
    hdr.table_capacity = table_capacity;
    hdr.sso_token_count = sso_count;
    hdr.ctrl_offset = header_padded_bytes;
    hdr.slots_offset = hdr.ctrl_offset + ctrl_padded_bytes;
    hdr.string_data_offset = hdr.slots_offset + slots_padded_bytes;
    hdr.string_data_bytes = total_heap_str_bytes;
    hdr.regex_offset = hdr.string_data_offset + string_padded_bytes;
    hdr.regex_bytes = regex_raw_len;
    hdr.specials_offset = hdr.regex_offset + regex_padded_bytes;
    hdr.specials_count = effective_specials_count;
    hdr.eot_token_id = (eot_token_id > 0) ? eot_token_id : DEFAULT_EOT_TOKEN_ID;
    hdr.fast_tables_magic = VIZTOK_FAST_MAGIC;
    hdr.byte_ranks_offset = hdr.specials_offset + specials_padded_bytes;
    hdr.pair_ranks_offset = hdr.byte_ranks_offset + byte_ranks_padded_bytes;
    hdr.id_to_slot_offset = hdr.pair_ranks_offset + pair_ranks_padded_bytes;

    char tmp_bin_path[2048];
    snprintf(tmp_bin_path, sizeof(tmp_bin_path), "%s.tmp", out_bin_path);
#if defined(_WIN32)
    FILE *f_out = open_file_utf8(tmp_bin_path, L"wb");
#else
    FILE *f_out = open_file_utf8(tmp_bin_path, "wb");
#endif
    if (!f_out) {
        free(ctrl);
        free(slots);
        if (str_blob) free(str_blob);
        free(local_specials);
        free(pre_byte_ranks);
        free(pre_pair_ranks);
        free(pre_id_to_slot);
        return -4;
    }

    uint8_t zero_pad[64] = {0};
    fwrite(&hdr, 1, sizeof(hdr), f_out);
    if (header_padded_bytes > sizeof(hdr)) {
        fwrite(zero_pad, 1, (size_t)(header_padded_bytes - sizeof(hdr)), f_out);
    }
    fwrite(ctrl, 1, (size_t)ctrl_padded_bytes, f_out);
    fwrite(slots, sizeof(ttkn_slot_t), table_capacity, f_out);
    if (slots_padded_bytes > (uint64_t)table_capacity * sizeof(ttkn_slot_t)) {
        fwrite(zero_pad, 1, (size_t)(slots_padded_bytes - (uint64_t)table_capacity * sizeof(ttkn_slot_t)), f_out);
    }
    if (total_heap_str_bytes > 0 && str_blob) {
        fwrite(str_blob, 1, (size_t)total_heap_str_bytes, f_out);
    }
    if (string_padded_bytes > total_heap_str_bytes) {
        fwrite(zero_pad, 1, (size_t)(string_padded_bytes - total_heap_str_bytes), f_out);
    }
    fwrite(chosen_regex, 1, regex_raw_len, f_out);
    if (regex_padded_bytes > regex_raw_len) {
        fwrite(zero_pad, 1, (size_t)(regex_padded_bytes - regex_raw_len), f_out);
    }
    fwrite(local_specials, sizeof(viztoken_special_disk_t), effective_specials_count, f_out);
    if (specials_padded_bytes > specials_raw_bytes) {
        fwrite(zero_pad, 1, (size_t)(specials_padded_bytes - specials_raw_bytes), f_out);
    }
    fwrite(pre_byte_ranks, 1, (size_t)byte_ranks_raw_bytes, f_out);
    if (byte_ranks_padded_bytes > byte_ranks_raw_bytes) {
        fwrite(zero_pad, 1, (size_t)(byte_ranks_padded_bytes - byte_ranks_raw_bytes), f_out);
    }
    fwrite(pre_pair_ranks, 1, (size_t)pair_ranks_raw_bytes, f_out);
    if (pair_ranks_padded_bytes > pair_ranks_raw_bytes) {
        fwrite(zero_pad, 1, (size_t)(pair_ranks_padded_bytes - pair_ranks_raw_bytes), f_out);
    }
    fwrite(pre_id_to_slot, 1, (size_t)id_to_slot_raw_bytes, f_out);
    if (id_to_slot_padded_bytes > id_to_slot_raw_bytes) {
        fwrite(zero_pad, 1, (size_t)(id_to_slot_padded_bytes - id_to_slot_raw_bytes), f_out);
    }

    flush_and_commit_file(f_out);
    fclose(f_out);
    atomic_replace_file_utf8(tmp_bin_path, out_bin_path);

    free(ctrl);
    free(slots);
    if (str_blob) free(str_blob);
    free(local_specials);
    free(pre_byte_ranks);
    free(pre_pair_ranks);
    free(pre_id_to_slot);

    clock_t t1 = clock();
    double elapsed_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;

    viztoken_telemetry_t telem = {
        .total_tokens = count,
        .table_capacity = table_capacity,
        .sso_tokens = sso_count,
        .max_psl = max_probe_groups,
        .string_blob_bytes = total_heap_str_bytes,
        .load_factor = (double)count / (double)table_capacity,
        .sso_ratio = count > 0 ? ((double)sso_count / (double)count) : 0.0,
        .mean_psl = mean_probe,
        .psl_variance = probe_var,
        .compile_seconds = elapsed_sec
    };
    if (out_telemetry) {
        *out_telemetry = telem;
    }
    if (out_telemetry_json && out_telemetry_json[0] != '\0') {
        char tmp_json_path[2048];
        snprintf(tmp_json_path, sizeof(tmp_json_path), "%s.tmp", out_telemetry_json);
#if defined(_WIN32)
        FILE *f_telem = open_file_utf8(tmp_json_path, L"wb");
#else
        FILE *f_telem = open_file_utf8(tmp_json_path, "wb");
#endif
        if (f_telem) {
            fprintf(f_telem, "{\n");
            fprintf(f_telem, "  \"total_tokens\": %u,\n", telem.total_tokens);
            fprintf(f_telem, "  \"table_capacity\": %u,\n", telem.table_capacity);
            fprintf(f_telem, "  \"load_factor\": %.6f,\n", telem.load_factor);
            fprintf(f_telem, "  \"sso_tokens\": %u,\n", telem.sso_tokens);
            fprintf(f_telem, "  \"sso_ratio\": %.6f,\n", telem.sso_ratio);
            fprintf(f_telem, "  \"string_blob_bytes\": %llu,\n", (unsigned long long)telem.string_blob_bytes);
            fprintf(f_telem, "  \"max_psl\": %u,\n", telem.max_psl);
            fprintf(f_telem, "  \"mean_psl\": %.6f,\n", telem.mean_psl);
            fprintf(f_telem, "  \"psl_variance\": %.6f,\n", telem.psl_variance);
            fprintf(f_telem, "  \"compile_seconds\": %.6f\n", telem.compile_seconds);
            fprintf(f_telem, "}\n");
            flush_and_commit_file(f_telem);
            fclose(f_telem);
            atomic_replace_file_utf8(tmp_json_path, out_telemetry_json);
        }
    }
    return 0;
}

VIZTOKEN_API int32_t viztoken_compile_vocab_v4(
    const char *in_tiktoken_path,
    const char *out_bin_path,
    const char *regex_pattern,
    const viztoken_special_disk_t *specials,
    uint32_t specials_count,
    uint32_t eot_token_id,
    const char *out_telemetry_json,
    viztoken_telemetry_t *out_telemetry
) {
    return viztoken_compile_vocab_v5(
        in_tiktoken_path,
        out_bin_path,
        regex_pattern,
        specials,
        specials_count,
        eot_token_id,
        out_telemetry_json,
        out_telemetry
    );
}

VIZTOKEN_API int32_t viztoken_compile_vocab_with_telemetry(
    const char *in_tiktoken_path,
    const char *out_bin_path,
    const char *out_telemetry_json,
    viztoken_telemetry_t *out_telemetry
) {
    return viztoken_compile_vocab_v5(
        in_tiktoken_path,
        out_bin_path,
        NULL,
        NULL,
        0,
        DEFAULT_EOT_TOKEN_ID,
        out_telemetry_json,
        out_telemetry
    );
}

VIZTOKEN_API int32_t viztoken_compile_vocab(const char *in_tiktoken_path, const char *out_bin_path) {
    return viztoken_compile_vocab_with_telemetry(in_tiktoken_path, out_bin_path, NULL, NULL);
}

#if defined(_WIN32)
static inline uint64_t next_regex_uid(void) {
    static volatile LONG64 seq = 1;
    return (uint64_t)InterlockedIncrement64(&seq);
}
#else
static inline uint64_t next_regex_uid(void) {
    static volatile uint64_t seq = 1;
    return __sync_add_and_fetch(&seq, 1);
}
#endif

static ttkn_regex_t *ttkn_regex_init(const char *pattern) {
    ttkn_regex_t *re = (ttkn_regex_t *)malloc(sizeof(ttkn_regex_t));
    if (!re) return NULL;
    int errcode = 0;
    PCRE2_SIZE erroffset = 0;
    uint32_t options = PCRE2_UTF | PCRE2_UCP;
    const char *actual_pat = (pattern && pattern[0] != '\0') ? pattern : DEFAULT_QWEN_PATTERN;
    re->code = pcre2_compile(
        (PCRE2_SPTR)actual_pat,
        PCRE2_ZERO_TERMINATED,
        options,
        &errcode,
        &erroffset,
        NULL
    );
    if (!re->code) {
        free(re);
        return NULL;
    }
    pcre2_jit_compile(re->code, PCRE2_JIT_COMPLETE);
    re->uid = next_regex_uid();
    return re;
}

static void ttkn_regex_destroy(ttkn_regex_t *re) {
    if (!re) return;
    if (re->code) pcre2_code_free(re->code);
    free(re);
}

static inline pcre2_match_data *get_tls_match_data(const ttkn_regex_t *re) {
#if defined(_MSC_VER)
    static __declspec(thread) pcre2_match_data *tls_md = NULL;
    static __declspec(thread) uint64_t tls_last_uid = 0;
#else
    static __thread pcre2_match_data *tls_md = NULL;
    static __thread uint64_t tls_last_uid = 0;
#endif
    if (!re || !re->code) return NULL;
    if (!tls_md || tls_last_uid != re->uid) {
        if (tls_md) pcre2_match_data_free(tls_md);
        tls_md = pcre2_match_data_create_from_pattern(re->code, NULL);
        tls_last_uid = re->uid;
    }
    return tls_md;
}

typedef struct {
    uint32_t byte_offset;
    uint32_t byte_len;
    int32_t  prev;
    int32_t  next;
    uint32_t pair_rank;
    uint32_t token_id;
    uint32_t gen;
} bpe_node_t;

typedef struct {
    uint32_t rank;
    int32_t  left_node;
    uint32_t gen;
} heap_item_t;

static inline void heap_push(heap_item_t *heap, uint32_t *size, uint32_t rank, int32_t left, uint32_t gen) {
    uint32_t i = (*size)++;
    while (i > 0) {
        uint32_t p = (i - 1) >> 1;
        if (heap[p].rank < rank || (heap[p].rank == rank && heap[p].left_node <= left)) break;
        heap[i] = heap[p];
        i = p;
    }
    heap[i].rank = rank;
    heap[i].left_node = left;
    heap[i].gen = gen;
}

static inline int heap_pop(heap_item_t *heap, uint32_t *size, heap_item_t *out) {
    if (*size == 0) return 0;
    *out = heap[0];
    heap_item_t last = heap[--(*size)];
    if (*size == 0) return 1;
    uint32_t i = 0;
    while ((i << 1) + 1 < *size) {
        uint32_t left = (i << 1) + 1;
        uint32_t right = left + 1;
        uint32_t best = left;
        if (right < *size) {
            if (heap[right].rank < heap[left].rank ||
               (heap[right].rank == heap[left].rank && heap[right].left_node < heap[left].left_node)) {
                best = right;
            }
        }
        if (last.rank < heap[best].rank ||
           (last.rank == heap[best].rank && last.left_node <= heap[best].left_node)) {
            break;
        }
        heap[i] = heap[best];
        i = best;
    }
    heap[i] = last;
    return 1;
}

static inline uint32_t eval_pair_rank(
    const viztoken_ctx_t *ctx,
    const uint8_t *base,
    const bpe_node_t *nodes,
    int32_t l_idx
) {
    int32_t r_idx = nodes[l_idx].next;
    if (r_idx < 0) return INF_RANK;
    uint32_t total_len = nodes[l_idx].byte_len + nodes[r_idx].byte_len;
    return ttkn_query_pair_rank(ctx, base, nodes[l_idx].byte_offset, total_len);
}

static inline int32_t ttkn_bpe_flat_merge(
    const viztoken_ctx_t *ctx,
    const uint8_t *piece,
    uint32_t len,
    int32_t *out_ids,
    uint32_t max_out
) {
    ttkn_part_t parts[FLAT_MERGE_MAX_LEN + 2];
    uint32_t min_rank = INF_RANK;
    uint32_t min_idx = 0;

    for (uint32_t i = 0; i < len - 1; i++) {
        uint16_t key;
        memcpy(&key, piece + i, 2);
        uint32_t r = ctx->pair_ranks_2byte[key];
        parts[i].offset = i;
        parts[i].rank = r;
        parts[i].token_id = ctx->byte_ranks[piece[i]];
        bool smaller = (r < min_rank);
        min_rank = smaller ? r : min_rank;
        min_idx  = smaller ? i : min_idx;
    }
    parts[len - 1].offset = len - 1;
    parts[len - 1].rank = INF_RANK;
    parts[len - 1].token_id = ctx->byte_ranks[piece[len - 1]];
    parts[len].offset = len;
    parts[len].rank = INF_RANK;
    parts[len].token_id = INF_RANK;

    uint32_t num_parts = len + 1;
    while (min_rank != INF_RANK) {
        uint32_t i = min_idx;
        parts[i].token_id = min_rank;
        if (i > 0) {
            if (i + 2 < num_parts) {
                uint32_t p_start = parts[i - 1].offset;
                uint32_t p_len = parts[i + 2].offset - p_start;
                parts[i - 1].rank = ttkn_query_pair_rank(ctx, piece, p_start, p_len);
            } else {
                parts[i - 1].rank = INF_RANK;
            }
        }
        if (i + 3 < num_parts) {
            uint32_t p_start = parts[i].offset;
            uint32_t p_len = parts[i + 3].offset - p_start;
            parts[i].rank = ttkn_query_pair_rank(ctx, piece, p_start, p_len);
        } else {
            parts[i].rank = INF_RANK;
        }
        for (uint32_t j = i + 1; j < num_parts - 1; j++) {
            parts[j] = parts[j + 1];
        }
        num_parts--;
        min_rank = INF_RANK;
        min_idx = 0;
        for (uint32_t j = 0; j < num_parts - 1; j++) {
            uint32_t r = parts[j].rank;
            bool smaller = (r < min_rank);
            min_rank = smaller ? r : min_rank;
            min_idx  = smaller ? j : min_idx;
        }
    }

    uint32_t emitted = 0;
    for (uint32_t j = 0; j < num_parts - 1 && emitted < max_out; j++) {
        out_ids[emitted++] = (int32_t)parts[j].token_id;
    }
    return (int32_t)emitted;
}

static inline int32_t ttkn_bpe_dynamic_heap(
    const viztoken_ctx_t *ctx,
    const uint8_t *piece,
    uint32_t len,
    int32_t *out_ids,
    uint32_t max_out
) {
    bpe_node_t *nodes = (bpe_node_t *)malloc(sizeof(bpe_node_t) * len);
    uint32_t heap_cap = len * 4;
    heap_item_t *heap = (heap_item_t *)malloc(sizeof(heap_item_t) * heap_cap);
    if (!nodes || !heap) {
        if (nodes) free(nodes);
        if (heap) free(heap);
        return 0;
    }

    uint32_t heap_size = 0;
    for (uint32_t i = 0; i < len; i++) {
        nodes[i].byte_offset = i;
        nodes[i].byte_len = 1;
        nodes[i].prev = (int32_t)i - 1;
        nodes[i].next = (i + 1 < len) ? (int32_t)(i + 1) : -1;
        nodes[i].pair_rank = INF_RANK;
        nodes[i].token_id = ctx->byte_ranks[piece[i]];
        nodes[i].gen = 0;
    }
    for (uint32_t i = 0; i < len - 1; i++) {
        uint32_t r = eval_pair_rank(ctx, piece, nodes, (int32_t)i);
        nodes[i].pair_rank = r;
        if (r != INF_RANK) {
            heap_push(heap, &heap_size, r, (int32_t)i, 0);
        }
    }

    while (heap_size > 0) {
        heap_item_t top;
        heap_pop(heap, &heap_size, &top);
        int32_t l = top.left_node;
        if (top.gen != nodes[l].gen || nodes[l].pair_rank != top.rank) continue;
        int32_t r = nodes[l].next;
        if (r < 0) continue;

        nodes[l].byte_len += nodes[r].byte_len;
        nodes[l].next = nodes[r].next;
        nodes[l].token_id = top.rank;
        if (nodes[r].next >= 0) {
            nodes[nodes[r].next].prev = l;
        }
        nodes[l].gen++;
        nodes[r].gen++;
        nodes[r].pair_rank = INF_RANK;
        nodes[r].next = -1;
        nodes[r].prev = -1;

        uint32_t r_new = eval_pair_rank(ctx, piece, nodes, l);
        nodes[l].pair_rank = r_new;
        if (r_new != INF_RANK) {
            heap_push(heap, &heap_size, r_new, l, nodes[l].gen);
        }
        if (nodes[l].prev >= 0) {
            int32_t p = nodes[l].prev;
            nodes[p].gen++;
            uint32_t p_new = eval_pair_rank(ctx, piece, nodes, p);
            nodes[p].pair_rank = p_new;
            if (p_new != INF_RANK) {
                heap_push(heap, &heap_size, p_new, p, nodes[p].gen);
            }
        }
    }

    uint32_t emitted = 0;
    int32_t cur = 0;
    while (cur >= 0 && emitted < max_out) {
        out_ids[emitted++] = (int32_t)nodes[cur].token_id;
        cur = nodes[cur].next;
    }
    free(nodes);
    free(heap);
    return (int32_t)emitted;
}

static inline int32_t ttkn_bpe_encode_chunk(
    const viztoken_ctx_t *ctx,
    const uint8_t *piece,
    uint32_t len,
    int32_t *out_ids,
    uint32_t max_out
) {
    if (!ctx || !piece || len == 0 || !out_ids || max_out == 0) return 0;
    if (len == 1) {
        out_ids[0] = (int32_t)ctx->byte_ranks[piece[0]];
        return 1;
    }
    if (len == 2) {
        uint16_t key;
        memcpy(&key, piece, 2);
        uint32_t r = ctx->pair_ranks_2byte[key];
        if (r != INF_RANK) {
            out_ids[0] = (int32_t)r;
            return 1;
        }
        out_ids[0] = (int32_t)ctx->byte_ranks[piece[0]];
        if (max_out > 1) {
            out_ids[1] = (int32_t)ctx->byte_ranks[piece[1]];
            return 2;
        }
        return 1;
    }

    if (len <= FLAT_MERGE_MAX_LEN) {
        uint64_t h = ttkn_hash64(piece, len);
        if (len <= 24 && len <= ctx->vm.max_token_len) {
            uint32_t whole_rank = INF_RANK;
            if (ttkn_swiss_lookup_with_hash(&ctx->vm, piece, len, h, &whole_rank)) {
                out_ids[0] = (int32_t)whole_rank;
                return 1;
            }
        }

        uint32_t c_idx = (uint32_t)((h >> 32) & (BPE_CACHE_SLOTS - 1U));
        bpe_cache_slot_t *c_slot = &tls_bpe_cache[c_idx];
        uint64_t uid = ctx->regex->uid;
        if (c_slot->hash == h &&
            c_slot->ctx_uid == uid &&
            c_slot->byte_len == (uint16_t)len &&
            c_slot->token_count <= max_out &&
            ttkn_memeq(c_slot->bytes, piece, len)) {
            uint32_t tc = c_slot->token_count;
            memcpy(out_ids, c_slot->tokens, (size_t)tc * sizeof(int32_t));
            return (int32_t)tc;
        }

        int32_t emitted = ttkn_bpe_flat_merge(ctx, piece, len, out_ids, max_out);
        if (emitted > 0 && (uint32_t)emitted <= BPE_CACHE_MAX_TOKENS && max_out >= len) {
            c_slot->hash = h;
            c_slot->ctx_uid = uid;
            c_slot->byte_len = (uint16_t)len;
            c_slot->token_count = (uint16_t)emitted;
            memcpy(c_slot->bytes, piece, len);
            memcpy(c_slot->tokens, out_ids, (size_t)emitted * sizeof(int32_t));
        }
        return emitted;
    }

    return ttkn_bpe_dynamic_heap(ctx, piece, len, out_ids, max_out);
}

typedef struct {
    const viztoken_ctx_t *ctx;
    int32_t *out_ids;
    uint32_t max_tokens;
    uint32_t count;
} encode_state_t;

static inline bool is_ascii_alpha(uint8_t c) {
    return (uint8_t)((c | 0x20U) - 'a') <= 25U;
}

static inline bool is_ascii_digit(uint8_t c) {
    return (uint8_t)(c - '0') <= 9U;
}

static int ttkn_hybrid_split_and_encode(
    const viztoken_ctx_t *ctx,
    pcre2_match_data *match_data,
    const uint8_t *utf8_text,
    uint32_t text_len,
    encode_state_t *st
) {
    PCRE2_SIZE start_offset = 0;
    uint32_t match_options = PCRE2_NOTEMPTY | PCRE2_NO_UTF_CHECK;
    const int fast_ascii = ctx->is_standard_qwen_regex;

    while (start_offset < (PCRE2_SIZE)text_len && st->count < st->max_tokens) {
        if (fast_ascii) {
            uint8_t c0 = utf8_text[start_offset];
            if (is_ascii_digit(c0)) {
                st->out_ids[st->count++] = (int32_t)ctx->byte_ranks[c0];
                start_offset++;
                continue;
            }
            if (is_ascii_alpha(c0) || (c0 == ' ' && start_offset + 1 < (PCRE2_SIZE)text_len && is_ascii_alpha(utf8_text[start_offset + 1]))) {
                PCRE2_SIZE end_pos = start_offset + 1;
                while (end_pos < (PCRE2_SIZE)text_len && is_ascii_alpha(utf8_text[end_pos])) {
                    end_pos++;
                }
                if (end_pos == (PCRE2_SIZE)text_len || utf8_text[end_pos] < 0x80U) {
                    uint32_t piece_len = (uint32_t)(end_pos - start_offset);
                    int32_t emitted = ttkn_bpe_encode_chunk(
                        ctx,
                        utf8_text + start_offset,
                        piece_len,
                        st->out_ids + st->count,
                        st->max_tokens - st->count
                    );
                    st->count += (uint32_t)emitted;
                    start_offset = end_pos;
                    continue;
                }
            }
        }

        int rc = pcre2_jit_match(
            ctx->regex->code,
            (PCRE2_SPTR)utf8_text,
            (PCRE2_SIZE)text_len,
            start_offset,
            match_options,
            match_data,
            NULL
        );
        if (rc < 0) {
            int32_t emitted = ttkn_bpe_encode_chunk(
                ctx,
                utf8_text + start_offset,
                text_len - (uint32_t)start_offset,
                st->out_ids + st->count,
                st->max_tokens - st->count
            );
            st->count += (uint32_t)emitted;
            break;
        }

        PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE m_start = ovector[0];
        PCRE2_SIZE m_end = ovector[1];
        if (m_start > start_offset && st->count < st->max_tokens) {
            int32_t emitted = ttkn_bpe_encode_chunk(
                ctx,
                utf8_text + start_offset,
                (uint32_t)(m_start - start_offset),
                st->out_ids + st->count,
                st->max_tokens - st->count
            );
            st->count += (uint32_t)emitted;
        }
        if (m_end > m_start && st->count < st->max_tokens) {
            int32_t emitted = ttkn_bpe_encode_chunk(
                ctx,
                utf8_text + m_start,
                (uint32_t)(m_end - m_start),
                st->out_ids + st->count,
                st->max_tokens - st->count
            );
            st->count += (uint32_t)emitted;
            start_offset = m_end;
        } else {
            start_offset++;
        }
    }
    return 0;
}

static int compare_specials_desc(const void *a, const void *b) {
    const special_entry_t *sa = (const special_entry_t *)a;
    const special_entry_t *sb = (const special_entry_t *)b;
    if (sb->len != sa->len) {
        return (sb->len > sa->len) ? 1 : -1;
    }
    return 0;
}

static inline int check_special_token(
    const viztoken_ctx_t *ctx,
    const char *ptr,
    uint32_t remaining,
    int32_t *out_id,
    uint32_t *out_len,
    const char **out_lit
) {
    for (uint32_t i = 0; i < ctx->special_count; i++) {
        uint32_t slen = ctx->specials[i].len;
        if (remaining >= slen && ttkn_memeq((const uint8_t *)ptr, (const uint8_t *)ctx->specials[i].str, slen)) {
            *out_id = ctx->specials[i].id;
            *out_len = slen;
            *out_lit = ctx->specials[i].str;
            return 1;
        }
    }
    return 0;
}

static inline uint32_t find_next_lt_simd(const char *text, uint32_t cursor, uint32_t text_bytes) {
#if defined(TTKN_HAS_AVX2)
    __m256i v_lt = _mm256_set1_epi8('<');
    while (cursor + 32 <= text_bytes) {
        __m256i chunk = _mm256_loadu_si256((const __m256i *)(text + cursor));
        __m256i cmp = _mm256_cmpeq_epi8(chunk, v_lt);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(cmp);
        if (mask != 0) {
            return cursor + ttkn_ctz32(mask);
        }
        cursor += 32;
    }
#elif defined(TTKN_HAS_SSE2)
    __m128i v_lt = _mm_set1_epi8('<');
    while (cursor + 16 <= text_bytes) {
        __m128i chunk = _mm_loadu_si128((const __m128i *)(text + cursor));
        __m128i cmp = _mm_cmpeq_epi8(chunk, v_lt);
        uint32_t mask = (uint32_t)_mm_movemask_epi8(cmp) & 0xFFFFU;
        if (mask != 0) {
            return cursor + ttkn_ctz32(mask);
        }
        cursor += 16;
    }
#endif
    while (cursor < text_bytes && text[cursor] != '<') {
        cursor++;
    }
    return cursor;
}

VIZTOKEN_API uint32_t viztoken_find_common_prefix(
    const int32_t *tokens_a,
    uint32_t len_a,
    const int32_t *tokens_b,
    uint32_t len_b
) {
    if (!tokens_a || !tokens_b) return 0;
    uint32_t limit = (len_a < len_b) ? len_a : len_b;
    uint32_t i = 0;
#if defined(TTKN_HAS_AVX2)
    while (i + 8 <= limit) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(tokens_a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(tokens_b + i));
        __m256i eq = _mm256_cmpeq_epi32(va, vb);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(eq);
        if (mask != 0xFFFFFFFFU) {
            return i + (ttkn_ctz32(~mask) >> 2);
        }
        i += 8;
    }
#endif
#if defined(TTKN_HAS_SSE2)
    while (i + 4 <= limit) {
        __m128i va = _mm_loadu_si128((const __m128i *)(tokens_a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(tokens_b + i));
        __m128i eq = _mm_cmpeq_epi32(va, vb);
        uint32_t mask = (uint32_t)_mm_movemask_epi8(eq) & 0xFFFFU;
        if (mask != 0xFFFFU) {
            return i + (ttkn_ctz32(~mask) >> 2);
        }
        i += 4;
    }
#endif
    while (i < limit) {
        if (tokens_a[i] != tokens_b[i]) {
            return i;
        }
        i++;
    }
    return limit;
}

VIZTOKEN_API viztoken_ctx_t *viztoken_init(const char *ranks_bin_path) {
    viztoken_ctx_t *ctx = (viztoken_ctx_t *)calloc(1, sizeof(viztoken_ctx_t));
    if (!ctx) return NULL;

    if (ttkn_mmap_open(&ctx->vm, ranks_bin_path) != 0) {
        free(ctx);
        return NULL;
    }

    ctx->base_vocab_size = ctx->vm.vocab_size;
    ctx->total_vocab_size = ctx->vm.total_vocab_size;
    ctx->eot_token_id = ctx->vm.eot_token_id;
    ctx->regex = ttkn_regex_init(ctx->vm.regex_pattern);
    if (!ctx->regex) {
        ttkn_mmap_close(&ctx->vm);
        free(ctx);
        return NULL;
    }

    const char *pat = ctx->vm.regex_pattern ? ctx->vm.regex_pattern : DEFAULT_QWEN_PATTERN;
    ctx->is_standard_qwen_regex = (
        strcmp(pat, DEFAULT_QWEN_PATTERN) == 0 ||
        strcmp(pat, DEFAULT_QWEN35_PATTERN) == 0
    ) ? 1 : 0;

    uint32_t max_id_tracker = ctx->total_vocab_size;
    if (ctx->vm.specials && ctx->vm.specials_count > 0) {
        ctx->specials = (special_entry_t *)malloc(sizeof(special_entry_t) * ctx->vm.specials_count);
        ctx->special_count = ctx->vm.specials_count;
        for (uint32_t i = 0; i < ctx->vm.specials_count; i++) {
            strncpy(ctx->specials[i].str, ctx->vm.specials[i].literal, 63);
            ctx->specials[i].str[63] = '\0';
            ctx->specials[i].len = (uint32_t)strlen(ctx->specials[i].str);
            ctx->specials[i].id = ctx->vm.specials[i].id;
            if ((uint32_t)ctx->specials[i].id > max_id_tracker) {
                max_id_tracker = (uint32_t)ctx->specials[i].id;
            }
        }
        qsort(ctx->specials, ctx->special_count, sizeof(special_entry_t), compare_specials_desc);
    } else {
        viztoken_special_disk_t *defaults = (viztoken_special_disk_t *)malloc(sizeof(viztoken_special_disk_t) * 512);
        uint32_t def_count = populate_default_qwen_specials(defaults);
        ctx->specials = (special_entry_t *)malloc(sizeof(special_entry_t) * def_count);
        ctx->special_count = def_count;
        for (uint32_t i = 0; i < def_count; i++) {
            strncpy(ctx->specials[i].str, defaults[i].literal, 63);
            ctx->specials[i].str[63] = '\0';
            ctx->specials[i].len = defaults[i].length;
            ctx->specials[i].id = defaults[i].id;
            if ((uint32_t)defaults[i].id > max_id_tracker) {
                max_id_tracker = (uint32_t)defaults[i].id;
            }
        }
        free(defaults);
        qsort(ctx->specials, ctx->special_count, sizeof(special_entry_t), compare_specials_desc);
    }

    memset(ctx->special_prefix_mask, 0, 256);
    int all_lt = 1;
    for (uint32_t i = 0; i < ctx->special_count; i++) {
        if (ctx->specials[i].len > 0) {
            uint8_t c = (uint8_t)ctx->specials[i].str[0];
            ctx->special_prefix_mask[c] = 1;
            if (c != '<') all_lt = 0;
        }
    }
    ctx->all_specials_start_with_lt = all_lt;
    ctx->max_allocated_id = max_id_tracker + 1;

    if (ctx->vm.mapped_byte_ranks && ctx->vm.mapped_pair_ranks && ctx->vm.mapped_id_to_slot) {
        ctx->byte_ranks = ctx->vm.mapped_byte_ranks;
        ctx->pair_ranks_2byte = ctx->vm.mapped_pair_ranks;
        ctx->id_to_slot = ctx->vm.mapped_id_to_slot;
        return ctx;
    }

    ctx->id_to_str_fallback = (char **)calloc(ctx->max_allocated_id, sizeof(char *));
    ctx->id_to_len_fallback = (uint32_t *)calloc(ctx->max_allocated_id, sizeof(uint32_t));
    ctx->owned_byte_ranks = (uint32_t *)malloc(256 * sizeof(uint32_t));
    ctx->owned_pair_ranks = (uint32_t *)malloc(65536 * sizeof(uint32_t));

    const ttkn_slot_t *slots = ctx->vm.slots;
    const uint8_t *ctrl = ctx->vm.ctrl;
    uint32_t cap = ctx->vm.table_capacity;
    for (uint32_t i = 0; i < cap; i++) {
        if (ctrl[i] != CTRL_EMPTY && ctrl[i] != CTRL_SENTINEL && slots[i].rank < ctx->max_allocated_id) {
            uint32_t r = slots[i].rank;
            if (slots[i].is_inline) {
                ctx->id_to_str_fallback[r] = (char *)slots[i].storage.inline_bytes;
            } else {
                ctx->id_to_str_fallback[r] = (char *)(ctx->vm.string_data + (size_t)slots[i].storage.heap.offset);
            }
            ctx->id_to_len_fallback[r] = slots[i].length;
        }
    }
    for (uint32_t i = 0; i < ctx->special_count; i++) {
        int32_t sid = ctx->specials[i].id;
        if (sid >= 0 && (uint32_t)sid < ctx->max_allocated_id) {
            ctx->id_to_str_fallback[sid] = ctx->specials[i].str;
            ctx->id_to_len_fallback[sid] = ctx->specials[i].len;
        }
    }
    for (uint32_t b = 0; b < 256; b++) {
        uint8_t byte_val = (uint8_t)b;
        uint32_t r = INF_RANK;
        ttkn_swiss_lookup(&ctx->vm, &byte_val, 1, &r);
        ctx->owned_byte_ranks[b] = r;
    }
    for (uint32_t i = 0; i < 65536; i++) {
        ctx->owned_pair_ranks[i] = INF_RANK;
    }
    for (uint32_t b0 = 0; b0 < 256; b0++) {
        for (uint32_t b1 = 0; b1 < 256; b1++) {
            uint8_t pair[2] = { (uint8_t)b0, (uint8_t)b1 };
            uint16_t key;
            memcpy(&key, pair, 2);
            uint32_t r = INF_RANK;
            if (ttkn_swiss_lookup(&ctx->vm, pair, 2, &r)) {
                ctx->owned_pair_ranks[key] = r;
            }
        }
    }
    ctx->byte_ranks = ctx->owned_byte_ranks;
    ctx->pair_ranks_2byte = ctx->owned_pair_ranks;
    return ctx;
}

VIZTOKEN_API int32_t viztoken_encode(
    viztoken_ctx_t *ctx,
    const char *text,
    uint32_t text_bytes,
    int32_t *out_token_ids,
    uint32_t max_tokens
) {
    if (!ctx || !text || text_bytes == 0 || !out_token_ids || max_tokens == 0) return 0;
    pcre2_match_data *match_data = get_tls_match_data(ctx->regex);
    if (!match_data) return 0;

    encode_state_t st = {
        .ctx = ctx,
        .out_ids = out_token_ids,
        .max_tokens = max_tokens,
        .count = 0
    };

    uint32_t cursor = 0;
    uint32_t segment_start = 0;

    if (ctx->all_specials_start_with_lt) {
        while (cursor < text_bytes && st.count < max_tokens) {
            cursor = find_next_lt_simd(text, cursor, text_bytes);
            if (cursor >= text_bytes) break;

            int32_t sp_id = 0;
            uint32_t sp_len = 0;
            const char *sp_lit = NULL;
            if (check_special_token(ctx, text + cursor, text_bytes - cursor, &sp_id, &sp_len, &sp_lit)) {
                if (cursor > segment_start) {
                    ttkn_hybrid_split_and_encode(
                        ctx,
                        match_data,
                        (const uint8_t *)text + segment_start,
                        cursor - segment_start,
                        &st
                    );
                }
                while (cursor + sp_len <= text_bytes && st.count < max_tokens &&
                       ttkn_memeq((const uint8_t *)(text + cursor), (const uint8_t *)sp_lit, sp_len)) {
                    st.out_ids[st.count++] = sp_id;
                    cursor += sp_len;
                }
                segment_start = cursor;
                continue;
            }
            cursor++;
        }
    } else {
        while (cursor < text_bytes && st.count < max_tokens) {
            uint8_t c = (uint8_t)text[cursor];
            if (ctx->special_prefix_mask[c]) {
                int32_t sp_id = 0;
                uint32_t sp_len = 0;
                const char *sp_lit = NULL;
                if (check_special_token(ctx, text + cursor, text_bytes - cursor, &sp_id, &sp_len, &sp_lit)) {
                    if (cursor > segment_start) {
                        ttkn_hybrid_split_and_encode(
                            ctx,
                            match_data,
                            (const uint8_t *)text + segment_start,
                            cursor - segment_start,
                            &st
                        );
                    }
                    while (cursor + sp_len <= text_bytes && st.count < max_tokens &&
                           ttkn_memeq((const uint8_t *)(text + cursor), (const uint8_t *)sp_lit, sp_len)) {
                        st.out_ids[st.count++] = sp_id;
                        cursor += sp_len;
                    }
                    segment_start = cursor;
                    continue;
                }
            }
            cursor++;
        }
    }

    if (cursor > segment_start && st.count < max_tokens) {
        ttkn_hybrid_split_and_encode(
            ctx,
            match_data,
            (const uint8_t *)text + segment_start,
            cursor - segment_start,
            &st
        );
    }
    return (int32_t)st.count;
}

VIZTOKEN_API int32_t viztoken_encode_ordinary(
    viztoken_ctx_t *ctx,
    const char *text,
    uint32_t text_bytes,
    int32_t *out_token_ids,
    uint32_t max_tokens
) {
    if (!ctx || !text || text_bytes == 0 || !out_token_ids || max_tokens == 0) return 0;
    pcre2_match_data *match_data = get_tls_match_data(ctx->regex);
    if (!match_data) return 0;

    encode_state_t st = {
        .ctx = ctx,
        .out_ids = out_token_ids,
        .max_tokens = max_tokens,
        .count = 0
    };
    ttkn_hybrid_split_and_encode(
        ctx,
        match_data,
        (const uint8_t *)text,
        text_bytes,
        &st
    );
    return (int32_t)st.count;
}

VIZTOKEN_API int32_t viztoken_decode(
    viztoken_ctx_t *ctx,
    const int32_t *token_ids,
    uint32_t id_count,
    char *out_text,
    uint32_t max_bytes
) {
    if (!ctx || !token_ids || id_count == 0) return 0;
    uint32_t total_needed = 0;
    uint32_t written = 0;
    bool truncated = false;

    if (ctx->id_to_slot) {
        const ttkn_slot_t *slots = ctx->vm.slots;
        const uint8_t *str_data = ctx->vm.string_data;
        const viztoken_special_disk_t *disk_specs = ctx->vm.specials;
        uint32_t spec_cnt = ctx->vm.specials_count;
        uint32_t vocab_lim = ctx->vm.total_vocab_size;

        for (uint32_t i = 0; i < id_count; i++) {
            int32_t id = token_ids[i];
            if (id < 0 || (uint32_t)id >= vocab_lim) continue;
            uint32_t slot_val = ctx->id_to_slot[(uint32_t)id];
            if (slot_val == SLOT_NONE) continue;

            const char *s = NULL;
            uint32_t len = 0;
            if (slot_val & SLOT_SPECIAL_BIT) {
                uint32_t sp_idx = slot_val & ~SLOT_SPECIAL_BIT;
                if (disk_specs && sp_idx < spec_cnt) {
                    s = disk_specs[sp_idx].literal;
                    len = disk_specs[sp_idx].length;
                }
            } else if (slot_val < ctx->vm.table_capacity) {
                const ttkn_slot_t *sl = &slots[slot_val];
                len = sl->length;
                s = sl->is_inline
                    ? (const char *)sl->storage.inline_bytes
                    : (const char *)(str_data + (size_t)sl->storage.heap.offset);
            }
            if (s && len > 0) {
                if (out_text && max_bytes > 0 && !truncated) {
                    if (written + len < max_bytes) {
                        memcpy(out_text + written, s, len);
                        written += len;
                    } else {
                        truncated = true;
                    }
                }
                total_needed += len;
            }
        }
    } else {
        for (uint32_t i = 0; i < id_count; i++) {
            int32_t id = token_ids[i];
            if (id >= 0 && (uint32_t)id < ctx->max_allocated_id) {
                uint32_t len = ctx->id_to_len_fallback[id];
                const char *s = ctx->id_to_str_fallback[id];
                if (s && len > 0) {
                    if (out_text && max_bytes > 0 && !truncated) {
                        if (written + len < max_bytes) {
                            memcpy(out_text + written, s, len);
                            written += len;
                        } else {
                            truncated = true;
                        }
                    }
                    total_needed += len;
                }
            }
        }
    }

    if (out_text && max_bytes > 0) {
        out_text[written] = '\0';
    }
    return (int32_t)total_needed;
}

VIZTOKEN_API void viztoken_destroy(viztoken_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->regex) ttkn_regex_destroy(ctx->regex);
    ttkn_mmap_close(&ctx->vm);
    if (ctx->id_to_str_fallback) free(ctx->id_to_str_fallback);
    if (ctx->id_to_len_fallback) free(ctx->id_to_len_fallback);
    if (ctx->owned_byte_ranks) free(ctx->owned_byte_ranks);
    if (ctx->owned_pair_ranks) free(ctx->owned_pair_ranks);
    if (ctx->specials) free(ctx->specials);
    free(ctx);
}

/* Legacy C-ABI Wrappers */
VIZTOKEN_API tiktalkin_ctx_t *tiktalkin_init(const char *ranks_bin_path) {
    return viztoken_init(ranks_bin_path);
}

VIZTOKEN_API int32_t tiktalkin_encode(
    tiktalkin_ctx_t *ctx,
    const char *text,
    uint32_t text_bytes,
    int32_t *out_token_ids,
    uint32_t max_tokens
) {
    return viztoken_encode(ctx, text, text_bytes, out_token_ids, max_tokens);
}

VIZTOKEN_API int32_t tiktalkin_encode_ordinary(
    tiktalkin_ctx_t *ctx,
    const char *text,
    uint32_t text_bytes,
    int32_t *out_token_ids,
    uint32_t max_tokens
) {
    return viztoken_encode_ordinary(ctx, text, text_bytes, out_token_ids, max_tokens);
}

VIZTOKEN_API int32_t tiktalkin_decode(
    tiktalkin_ctx_t *ctx,
    const int32_t *token_ids,
    uint32_t id_count,
    char *out_text,
    uint32_t max_bytes
) {
    return viztoken_decode(ctx, token_ids, id_count, out_text, max_bytes);
}

VIZTOKEN_API uint32_t tiktalkin_find_common_prefix(
    const int32_t *tokens_a,
    uint32_t len_a,
    const int32_t *tokens_b,
    uint32_t len_b
) {
    return viztoken_find_common_prefix(tokens_a, len_a, tokens_b, len_b);
}

VIZTOKEN_API int32_t tiktalkin_compile_vocab(const char *in_tiktoken_path, const char *out_bin_path) {
    return viztoken_compile_vocab(in_tiktoken_path, out_bin_path);
}

VIZTOKEN_API int32_t tiktalkin_compile_vocab_with_telemetry(
    const char *in_tiktoken_path,
    const char *out_bin_path,
    const char *out_telemetry_json,
    ttkn_telemetry_t *out_telemetry
) {
    return viztoken_compile_vocab_with_telemetry(in_tiktoken_path, out_bin_path, out_telemetry_json, out_telemetry);
}

VIZTOKEN_API int32_t tiktalkin_compile_vocab_v4(
    const char *in_tiktoken_path,
    const char *out_bin_path,
    const char *regex_pattern,
    const ttkn_special_disk_t *specials,
    uint32_t specials_count,
    uint32_t eot_token_id,
    const char *out_telemetry_json,
    ttkn_telemetry_t *out_telemetry
) {
    return viztoken_compile_vocab_v5(
        in_tiktoken_path,
        out_bin_path,
        regex_pattern,
        specials,
        specials_count,
        eot_token_id,
        out_telemetry_json,
        out_telemetry
    );
}

VIZTOKEN_API void tiktalkin_destroy(tiktalkin_ctx_t *ctx) {
    viztoken_destroy(ctx);
}