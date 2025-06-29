// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>

#include "jsmn.h"

typedef struct {
    ngx_str_t prefill_location;
} ngx_http_prefill_loc_conf_t;

static ngx_int_t ngx_http_prefill_handler(ngx_http_request_t *r);
static void *ngx_http_prefill_create_loc_conf(ngx_conf_t *cf);
static char *ngx_http_prefill_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child);
static ngx_int_t ngx_http_prefill_init(ngx_conf_t *cf);

static ngx_command_t ngx_http_prefill_commands[] = {

    {ngx_string("prefill"),
        NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
        ngx_conf_set_str_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_prefill_loc_conf_t, prefill_location),
        NULL},

    ngx_null_command};

static ngx_http_module_t ngx_http_prefill_module_ctx = {
    NULL,                  /* preconfiguration */
    ngx_http_prefill_init, /* postconfiguration */

    NULL, /* create main configuration */
    NULL, /* init main configuration */

    NULL, /* create server configuration */
    NULL, /* merge server configuration */

    ngx_http_prefill_create_loc_conf, /* create location configuration */
    ngx_http_prefill_merge_loc_conf   /* merge location configuration */
};

ngx_module_t ngx_http_prefill_module = {NGX_MODULE_V1,
    &ngx_http_prefill_module_ctx, /* module context */
    ngx_http_prefill_commands,    /* module directives */
    NGX_HTTP_MODULE,              /* module type */
    NULL,                         /* init master */
    NULL,                         /* init module */
    NULL,                         /* init process */
    NULL,                         /* init thread */
    NULL,                         /* exit thread */
    NULL,                         /* exit process */
    NULL,                         /* exit master */
    NGX_MODULE_V1_PADDING};

typedef struct {
    ngx_uint_t done;
    ngx_uint_t status;
    ngx_uint_t response_complete;  // New flag to track response completion
    u_char *origin_body_data;
    ngx_uint_t origin_body_data_size;
    jsmntok_t *origin_body_tokens;
    int origin_body_tokens_size;
    u_char *prefill_response_body;
    ngx_uint_t prefill_response_body_size;
} ngx_http_prefill_ctx_t;

ngx_str_t ngx_concat_str(ngx_pool_t *pool, ngx_str_t s1, ngx_str_t s2)
{
    ngx_str_t result;

    result.len = s1.len + s2.len;
    result.data = ngx_palloc(pool, result.len);
    if (result.data == NULL) {
        result.len = 0;
        return result;
    }

    ngx_memcpy(result.data, s1.data, s1.len);
    ngx_memcpy(result.data + s1.len, s2.data, s2.len);

    return result;
}

// Helper: Find the index of a key in a JSMN token array
int find_jsmn_key(ngx_http_request_t *r, const char *json, jsmntok_t *tokens, int tokens_size, const char *key)
{
    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "gen decode request: token size %d", tokens_size);

    for (int i = 1; i < tokens_size; i++) {
        if (tokens[i].type == JSMN_STRING && (int)strlen(key) == tokens[i].end - tokens[i].start &&
            strncmp(json + tokens[i].start, key, tokens[i].end - tokens[i].start) == 0) {
            return i;
        }
    }
    return -1;
}

static char *prefill_response_json_keys[] = {
    "kv_transfer_params",
};

static unsigned int prefill_response_json_keys_len = sizeof(prefill_response_json_keys) / sizeof(char *);

static void ngx_http_gen_decode_request_body(ngx_http_request_t *r, ngx_http_prefill_ctx_t *ctx)
{
    // Parse prefill_response_body
    jsmn_parser parser;
    int tokens_size = 256;
    jsmntok_t *tokens = NULL;
    ngx_chain_t *chain_1st = NULL;
    ngx_chain_t *chain = NULL;
    ngx_buf_t *b = NULL;
    ngx_chain_t *chain_new = NULL;
    ngx_buf_t *b_new = NULL;
    int total_len = 0;

    tokens = ngx_palloc(r->pool, tokens_size * sizeof(jsmntok_t));
    if (!tokens) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode json: palloc jsmntok_t");
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP,
        r->connection->log,
        0,
        "gen decode request: prefill response body: %d %s",
        ctx->prefill_response_body_size,
        ctx->prefill_response_body);

    jsmn_init(&parser);
    int prefill_tokens_size =
        jsmn_parse(&parser, (char *)(ctx->prefill_response_body), ctx->prefill_response_body_size, tokens, tokens_size);
    while (prefill_tokens_size == JSMN_ERROR_NOMEM) {
        ngx_pfree(r->pool, tokens);
        tokens_size *= 2;
        jsmntok_t *new_tokens = ngx_palloc(r->pool, tokens_size * sizeof(jsmntok_t));
        if (!new_tokens) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode json: palloc new jsmntok_t %d", tokens_size);
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
        tokens = new_tokens;
        jsmn_init(&parser);
        prefill_tokens_size = jsmn_parse(
            &parser, (char *)(ctx->prefill_response_body), ctx->prefill_response_body_size, tokens, tokens_size);
    }

    // Create  chain for decode request
    chain = ngx_alloc_chain_link(r->pool);
    if (chain == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to allocate chain link");
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    // 1st buf in chain is origin body
    b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
    if (b == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to ngx_pcalloc");
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }
    b->pos = ctx->origin_body_data;
    b->last = b->pos + ctx->origin_body_data_size;
    b->memory = 1; /* content is in read-only memory */
    chain->buf = b;
    chain_1st = chain;

    // bufs then are from prefill response, one for each key
    for (int i = 0; i < (int)prefill_response_json_keys_len; i++) {
        int key_idx = find_jsmn_key(
            r, (char *)(ctx->prefill_response_body), tokens, prefill_tokens_size, prefill_response_json_keys[i]);
        if (key_idx == -1) {
            ngx_log_debug1(NGX_LOG_DEBUG_HTTP,
                r->connection->log,
                0,
                "gen decode request: key not found %s",
                prefill_response_json_keys[i]);
            continue;
        }

        while (b->last > b->pos) {
            if (b->last[0] == '}') {
                break;
            }
            b->last -= 1;
        }

        // Key found, we will palloc a new buf for this key value pair string
        int val_idx = key_idx + 1;
        int val_len = tokens[val_idx].end - tokens[val_idx].start;
        int key_len = ngx_strlen(prefill_response_json_keys[i]);
        int len = key_len + val_len + 16;
        b_new = ngx_create_temp_buf(r->pool, len);
        if (b_new == NULL) {
            ngx_log_error(
                NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to ngx_create_temp_buf %d", len);
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
        int pos = 0;
        b_new->pos[pos++] = ',';
        b_new->pos[pos++] = '\"';
        ngx_memcpy(b_new->pos + pos, prefill_response_json_keys[i], key_len);
        pos += ngx_strlen(prefill_response_json_keys[i]);
        b_new->pos[pos++] = '\"';
        b_new->pos[pos++] = ':';
        if (tokens[val_idx].type == JSMN_STRING) {
            b_new->pos[pos++] = '\"';
        }
        ngx_memcpy(b_new->pos + pos, ctx->prefill_response_body + tokens[val_idx].start, val_len);
        pos += val_len;
        if (tokens[val_idx].type == JSMN_STRING) {
            b_new->pos[pos++] = '\"';
        }
        b_new->pos[pos++] = '}';
        b_new->pos[pos] = '\0';
        b_new->last = b_new->pos + pos;
        b_new->memory = 1; /* content is in read-only memory */
        b_new->last_buf = 1;
        b_new->last_in_chain = 1;

        chain_new = ngx_alloc_chain_link(r->pool);
        if (chain == NULL) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to allocate new chain link");
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
        chain_new->buf = b_new;
        chain->next = chain_new;

        chain = chain_new;
        b = b_new;
    }

    b->last_buf = 1;
    b->last_in_chain = 1;
    chain->next = NULL;

    // Set up the subrequest's body structure
    if (r->request_body == NULL) {
        r->request_body = ngx_pcalloc(r->pool, sizeof(ngx_http_request_body_t));
        if (r->request_body == NULL) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to allocate request_body_t");
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
    }

    // Update subrequest's body properties
    r->request_body->bufs = chain_1st;
    r->request_body->buf = NULL;

    // Clear any existing temp file in subrequest
    if (r->request_body->temp_file) {
        r->request_body->temp_file = NULL;
    }

    for (ngx_chain_t *cl = r->request_body->bufs; cl; cl = cl->next) {
        total_len += ngx_buf_size(cl->buf);
    }

#if (NGX_DEBUG)
    char *json_str = ngx_pcalloc(r->pool, total_len + 1);
    int cur_pos = 0;
    for (ngx_chain_t *cl = r->request_body->bufs; cl; cl = cl->next) {
        ngx_memcpy(json_str + cur_pos, cl->buf->pos, ngx_buf_size(cl->buf));
        cur_pos += ngx_buf_size(cl->buf);
    }

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP,
        r->connection->log,
        0,
        "gen decode request: body for subrequest: %d %s",
        total_len,
        json_str);
#endif

    // Set content length in header
    r->headers_in.content_length_n = total_len;
    if (r->headers_in.content_length) {
        r->headers_in.content_length->value.len =
            ngx_sprintf(r->headers_in.content_length->value.data, "%uz", total_len) -
            r->headers_in.content_length->value.data;
    }

    return;
}

static ngx_int_t ngx_http_prefill_subrequest_done(ngx_http_request_t *r, void *data, ngx_int_t rc)
{
    ngx_http_prefill_ctx_t *ctx;
    ngx_chain_t *cl;
    size_t total = 0;
    u_char *p;

    ctx = (ngx_http_prefill_ctx_t *)data;
    if (rc != NGX_OK) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: subrequest failed with code %i", rc);
        ctx->done = 1;
        ctx->status = NGX_HTTP_INTERNAL_SERVER_ERROR;
        ngx_http_finalize_request(r->main, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }

    ctx->done = 1;
    ctx->status = r->headers_out.status;

    // Traverse the out_bufs chain to process the entire response body
    for (cl = r->out; cl; cl = cl->next) {
        total += ngx_buf_size(cl->buf);
    }

    ngx_log_error(
        NGX_LOG_INFO, r->connection->log, 0, "done prefill subrequest r:%p %d, status:%i", r, total, ctx->status);

    // Allocate memory for the temporary body copy + null terminator for cJSON
    ctx->prefill_response_body = ngx_palloc(r->main->pool, total + 1);
    if (ctx->prefill_response_body == NULL) {
        ngx_http_finalize_request(r, NGX_ERROR);
        ngx_log_error(
            NGX_LOG_ERR, r->connection->log, 0, "done prefill subrequest, malloc prefill response body buffer failed");
        return rc;
    }

    // Copy main request's buffer chain to a temporary contiguous block
    p = ctx->prefill_response_body;
    for (cl = r->out; cl; cl = cl->next) {
        size_t buf_size = ngx_buf_size(cl->buf);
        if (buf_size > 0) {
            p = ngx_cpymem(p, cl->buf->pos, buf_size);
        }
    }
    *p = '\0';
    ctx->prefill_response_body_size = total;

    if (ctx->status >= NGX_HTTP_OK && ctx->status < NGX_HTTP_SPECIAL_RESPONSE) {
        ngx_http_core_run_phases(r->main);  // status < 300: resume processing the main request
    } else {
        // subrequest status >= 300: set client response header from subrequest
        r->main->headers_out.status = ctx->status;
        r->main->headers_out.content_length_n = ctx->prefill_response_body_size;
        r->main->headers_out.content_type.data = r->headers_out.content_type.data;
        r->main->headers_out.content_type.len = r->headers_out.content_type.len;

        ngx_http_send_header(r->main);
    }
    return rc;
}

// Helper to copy a substring from JSON based on token
void json_token_tostr(const char *json, const jsmntok_t *t, char *buf, size_t buflen)
{
    size_t len = t->end - t->start;
    if (len >= buflen)
        len = buflen - 1;
    strncpy(buf, json + t->start, len);
    buf[len] = '\0';
}

// Info for all modifications, in the order they appear in the JSON
typedef enum { R_MAX_TOKENS, R_STREAM, R_STREAM_OPTIONS } region_type_t;
typedef struct {
    region_type_t type;
    size_t idx;    // For value keys: value token index; For stream_options: key token index
    size_t start;  // Start position in JSON for head (for stream_options it's the key start)
    size_t end;    // End position in JSON for tail (for stream_options it's the value end)
} region_info_t;

void gen_prefill_json_str_jsmn(
    ngx_http_request_t *r, ngx_http_prefill_ctx_t *ctx, const char *json, size_t len, char **out, size_t *out_len)
{
    jsmn_parser parser;
    int tokens_size = 256;
    jsmntok_t *tokens = NULL;
    int ret = -1;
    region_info_t region_infos[3];
    int region_infos_count = 0;
    int max_tokens_val_idx = -1;
    int stream_val_idx = -1;
    int stream_options_key_idx = -1, stream_options_val_idx = -1;
    char keybuf[64];

    tokens = ngx_palloc(r->pool, tokens_size * sizeof(jsmntok_t));
    if (!tokens) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen prefill json: palloc jsmntok_t");
        return;
    }

    jsmn_init(&parser);
    ret = jsmn_parse(&parser, json, len, tokens, tokens_size);
    while (ret == JSMN_ERROR_NOMEM) {
        ngx_pfree(r->pool, tokens);
        tokens_size *= 2;
        jsmntok_t *new_tokens = ngx_palloc(r->pool, tokens_size * sizeof(jsmntok_t));
        if (!new_tokens) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen prefill json: palloc new jsmntok_t %d", tokens_size);
            return;
        }
        tokens = new_tokens;
        jsmn_init(&parser);
        ret = jsmn_parse(&parser, json, len, tokens, tokens_size);
    }

    if (ret < 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen prefill json: failed to parse json using jsmn %d", ret);
        return;
    }

    ctx->origin_body_tokens = tokens;
    ctx->origin_body_tokens_size = ret;

    for (int i = 1; i < ret; ++i) {
        if (tokens[i].type == JSMN_STRING) {
            json_token_tostr(json, &tokens[i], keybuf, sizeof(keybuf));
            if (max_tokens_val_idx == -1 && strcmp(keybuf, "max_tokens") == 0) {
                max_tokens_val_idx = i + 1;
                region_infos[region_infos_count++] = (region_info_t){
                    R_MAX_TOKENS, max_tokens_val_idx, tokens[max_tokens_val_idx].start, tokens[max_tokens_val_idx].end};
            }
            if (stream_val_idx == -1 && strcmp(keybuf, "stream") == 0) {
                stream_val_idx = i + 1;
                region_infos[region_infos_count++] =
                    (region_info_t){R_STREAM, stream_val_idx, tokens[stream_val_idx].start, tokens[stream_val_idx].end};
            }
            if (stream_options_key_idx == -1 && strcmp(keybuf, "stream_options") == 0) {
                stream_options_key_idx = i;
                stream_options_val_idx = i + 1;
                // Locate key start and value end for removal, including preceding comma if any
                size_t so_key_start = tokens[stream_options_key_idx].start;
                size_t so_val_end = tokens[stream_options_val_idx].end;
                size_t j = so_key_start;
                while (j > 0 && json[j - 1] != ',')
                    j--;
                if (j > 0 && json[j - 1] == ',')
                    so_key_start = j - 1;
                region_infos[region_infos_count++] =
                    (region_info_t){R_STREAM_OPTIONS, stream_options_key_idx, so_key_start, so_val_end};
            }
        }
    }

    // Sort region_infos by start position (should be in order already due to scan, but robust)
    for (int i = 0; i < region_infos_count - 1; ++i) {
        for (int j = i + 1; j < region_infos_count; ++j) {
            if (region_infos[j].start < region_infos[i].start) {
                region_info_t tmp = region_infos[i];
                region_infos[i] = region_infos[j];
                region_infos[j] = tmp;
            }
        }
    }

    size_t cap = len + 64;  // ensure enough room, since we're only reducing/removing
    char *newjson = ngx_palloc(r->pool, cap);
    if (!newjson) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen prefill json: palooc %d", cap);
        return;
    }

    size_t pos = 0;
    size_t src = 0;  // source position in json to copy from

    for (int i = 0; i < region_infos_count; ++i) {
        region_info_t *ri = &region_infos[i];
        // Copy up to region
        if (ri->start > src) {
            memcpy(newjson + pos, json + src, ri->start - src);
            pos += ri->start - src;
        }
        switch (ri->type) {
            case R_MAX_TOKENS:
                memcpy(newjson + pos, "1", 1);
                pos += 1;
                src = ri->end;
                break;
            case R_STREAM:
                memcpy(newjson + pos, "false", 5);
                pos += 5;
                src = ri->end;
                break;
            case R_STREAM_OPTIONS:
                src = ri->end;
                break;
        }
    }
    // Copy the remainder before closing }
    if (src < len - 1) {
        memcpy(newjson + pos, json + src, len - 1 - src);
        pos += len - 1 - src;
    }

    // If "max_tokens" was missing, insert before '}'
    if (max_tokens_val_idx == -1) {
        if (pos > 0 && newjson[pos - 1] != '{') {
            newjson[pos++] = ',';
        }
        const char *insertion = "\"max_tokens\":1";
        memcpy(newjson + pos, insertion, strlen(insertion));
        pos += strlen(insertion);
    }

    // Add closing '}'
    newjson[pos++] = '}';
    newjson[pos] = '\0';
    *out = newjson;
    *out_len = pos;

    return;
}

static ngx_int_t ngx_http_gen_prefill_request_body(
    ngx_http_request_t *r, ngx_http_request_t *sr, ngx_http_prefill_ctx_t *ctx)
{
    ngx_chain_t *cl;
    size_t len = 0;
    u_char *body_data = NULL;
    u_char *p;
    char *modified_json_str = NULL;
    ngx_buf_t *b;

    if (r->request_body == NULL || r->request_body->bufs == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: request body is empty");
        return NGX_ERROR;
    }

    // Calculate total body size from the main request
    for (cl = r->request_body->bufs; cl != NULL; cl = cl->next) {
        ngx_buf_t *buf = cl->buf;
        if (buf->in_file) {
            len += (size_t)(buf->file_last - buf->file_pos);
        } else {
            len += ngx_buf_size(buf);
        }
    }

    if (len == 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: request body length is zero");
        return NGX_ERROR;
    }

    // Allocate memory for the temporary body copy + null terminator
    body_data = ngx_palloc(r->pool, len + 1);
    if (body_data == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: failed to allocate memory for body");
        return NGX_ERROR;
    }

    ctx->origin_body_data = body_data;  // record body data for decode request
    ctx->origin_body_data_size = len;

    // Copy main request's buffer chain to a temporary contiguous block
    p = body_data;
    for (cl = r->request_body->bufs; cl != NULL; cl = cl->next) {
        ngx_buf_t *buf = cl->buf;
        size_t buf_size;
        if (buf->in_file) {
            buf_size = (size_t)(buf->file_last - buf->file_pos);
            if (buf_size > 0) {
                ssize_t n = ngx_read_file(buf->file, p, buf_size, buf->file_pos);
                if (n != (ssize_t)buf_size) {
                    ngx_log_error(NGX_LOG_ERR,
                        r->connection->log,
                        0,
                        "prefill: failed to read body from file, expected %uz, got %z",
                        buf_size,
                        n);
                    return NGX_ERROR;
                }
                p += buf_size;
            }
        } else {
            buf_size = ngx_buf_size(buf);
            if (buf_size > 0) {
                p = ngx_cpymem(p, buf->pos, buf_size);
            }
        }
    }
    *p = '\0';

    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "prefill: original body for subrequest: %s", body_data);

    // Parse JSON from the temporary copy
    size_t str_len = 0;
    gen_prefill_json_str_jsmn(sr, ctx, (char *)body_data, len, &modified_json_str, &str_len);
    if (modified_json_str == NULL || str_len == 0) {
        return NGX_ERROR;
    }

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP,
        r->connection->log,
        0,
        "prefill: modified body for subrequest: %d, %s",
        str_len,
        modified_json_str);

    b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
    if (b == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: failed to ngx_pcalloc buf");
        return NGX_ERROR;
    }
    b->pos = (u_char *)modified_json_str;
    b->last = b->pos + str_len;
    b->memory = 1;
    b->last_buf = (sr->request_body_no_buffering) ? 0 : 1;
    b->last_in_chain = 1;

    // Create new chain for subrequest
    ngx_chain_t *new_chain = ngx_alloc_chain_link(sr->pool);
    if (new_chain == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: failed to allocate chain link");
        return NGX_ERROR;
    }
    new_chain->buf = b;
    new_chain->next = NULL;

    // Set up the subrequest's body structure
    if (sr->request_body == NULL) {
        sr->request_body = ngx_pcalloc(sr->pool, sizeof(ngx_http_request_body_t));
        if (sr->request_body == NULL) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: failed to allocate request_body_t");
            return NGX_ERROR;
        }
    }

    // Update subrequest's body properties
    sr->request_body->bufs = new_chain;
    sr->request_body->buf = b;
    sr->request_body->rest = 0;
    sr->request_body_in_file_only = 0;
    sr->request_body_in_persistent_file = 0;
    sr->request_body_in_clean_file = 0;

    // Clear any existing temp file in subrequest
    if (sr->request_body->temp_file) {
        sr->request_body->temp_file = NULL;
    }

    // Update subrequest's Content-Length header
    sr->headers_in.content_length_n = str_len;
    if (sr->headers_in.content_length) {
        sr->headers_in.content_length->value.len =
            ngx_sprintf(sr->headers_in.content_length->value.data, "%uz", str_len) -
            sr->headers_in.content_length->value.data;
    }
    sr->request_length = str_len;
    return NGX_DONE;
}

static void ngx_http_prefill_request_modify_handler(ngx_http_request_t *r)
{
    ngx_http_prefill_ctx_t *ctx;
    ngx_http_post_subrequest_t *ps;
    ngx_str_t uri;
    ngx_http_prefill_loc_conf_t *ulcf;
    ngx_http_request_t *sr;

    ctx = ngx_http_get_module_ctx(r, ngx_http_prefill_module);
    if (ctx == NULL) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    ps = ngx_palloc(r->pool, sizeof(ngx_http_post_subrequest_t));
    if (ps == NULL) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    ps->handler = ngx_http_prefill_subrequest_done;
    ps->data = ctx;

    // Create the URI for subrequest
    ulcf = ngx_http_get_module_loc_conf(r, ngx_http_prefill_module);
    if (ulcf == NULL) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }
    uri = ngx_concat_str(r->pool, ulcf->prefill_location, r->uri);
    if (uri.data == NULL) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    ngx_int_t flags = NGX_HTTP_SUBREQUEST_WAITED | NGX_HTTP_SUBREQUEST_IN_MEMORY;

    if (ngx_http_subrequest(r, &uri, &r->args, &sr, ps, flags) != NGX_OK) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    // Copy request method and headers
    sr->method = r->method;
    sr->method_name = r->method_name;

    // Copy request headers
    ngx_http_headers_in_t *headers_in = &sr->headers_in;
    headers_in->content_length_n = r->headers_in.content_length_n;
    headers_in->content_type = r->headers_in.content_type;

    // Set the request body for the subrequest, call ngx_http_finalize_request to count--,
    // otherwise connection under the hood will be left unclosed
    ngx_http_finalize_request(r, ngx_http_gen_prefill_request_body(r, sr, ctx));
}

static ngx_int_t ngx_http_prefill_handler(ngx_http_request_t *r)
{
    ngx_http_prefill_loc_conf_t *ulcf;
    ngx_http_prefill_ctx_t *ctx;
    ngx_chain_t out;
    ngx_buf_t *b;

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "http prefill handler");

    ulcf = ngx_http_get_module_loc_conf(r, ngx_http_prefill_module);

    if (ulcf->prefill_location.len == 0) {
        return NGX_DECLINED;
    }

    ctx = ngx_http_get_module_ctx(r, ngx_http_prefill_module);
    if (ctx != NULL) {
        if (!ctx->done) {
            return NGX_AGAIN;
        }
        if (ctx->status >= NGX_HTTP_OK && ctx->status < NGX_HTTP_SPECIAL_RESPONSE) {
            ngx_http_gen_decode_request_body(r, ctx);
            return NGX_OK;
        }

        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill unexpected status: %ui", ctx->status);

        // Output subrequest's response to client
        b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
        if (b == NULL) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: failed to ngx_pcalloc");
            ngx_http_finalize_request(r, NGX_ERROR);
            return NGX_HTTP_INTERNAL_SERVER_ERROR;
        }

        b->pos = ctx->prefill_response_body;
        b->last = ctx->prefill_response_body + ctx->prefill_response_body_size;
        b->memory = 1;
        b->last_buf = 1;
        out.buf = b;
        out.next = NULL;
        ngx_http_output_filter(r, &out);
        return NGX_ABORT;
    }

    ctx = ngx_pcalloc(r->pool, sizeof(ngx_http_prefill_ctx_t));
    if (ctx == NULL) {
        return NGX_ERROR;
    }

    ngx_http_set_ctx(r, ctx, ngx_http_prefill_module);

    // Read client request body
    ngx_int_t rc = ngx_http_read_client_request_body(r, ngx_http_prefill_request_modify_handler);
    if (rc >= NGX_HTTP_SPECIAL_RESPONSE) {
        return rc;
    }

    return NGX_AGAIN;
}

static void *ngx_http_prefill_create_loc_conf(ngx_conf_t *cf)
{
    ngx_http_prefill_loc_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_prefill_loc_conf_t));
    if (conf == NULL) {
        return NULL;
    }

    return conf;
}

static char *ngx_http_prefill_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_prefill_loc_conf_t *prev = parent;
    ngx_http_prefill_loc_conf_t *conf = child;

    ngx_conf_merge_str_value(conf->prefill_location, prev->prefill_location, "");

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_prefill_init(ngx_conf_t *cf)
{
    ngx_http_handler_pt *h;
    ngx_http_core_main_conf_t *cmcf;

    cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    h = ngx_array_push(&cmcf->phases[NGX_HTTP_ACCESS_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }

    *h = ngx_http_prefill_handler;

    return NGX_OK;
}