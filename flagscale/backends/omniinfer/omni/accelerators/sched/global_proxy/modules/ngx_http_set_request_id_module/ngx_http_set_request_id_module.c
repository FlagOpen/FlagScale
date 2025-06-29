// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>

#include <uuid/uuid.h>

#define UUID_STR_LEN 37 /* 36 for uuid, and 1 for '\0' */

static void gen_uuid(unsigned char out[UUID_STR_LEN])
{
    uuid_t uuid_data;
    uuid_generate(uuid_data);
    uuid_unparse_lower(uuid_data, (char *)out);
    return;
}

static u_char x_request_id[] = "X-Request-Id";

static ngx_int_t ngx_http_set_request_id_handler(ngx_http_request_t *r)
{
    u_char *p;
    ngx_table_elt_t *h;
    ngx_list_part_t *part;
    ngx_table_elt_t *header;
    ngx_uint_t i;

    if (r != r->main) {
        // Skip adding a new request id for subrequests
        return NGX_DECLINED;
    }

    // First, check if X-Request-Id already exists
    part = &r->headers_in.headers.part;
    header = part->elts;

    for (i = 0; /* void */; i++) {
        if (i >= part->nelts) {
            if (part->next == NULL) {
                break;
            }
            part = part->next;
            header = part->elts;
            i = 0;
        }

        if (header[i].key.len == sizeof(x_request_id) - 1 &&
            ngx_strncasecmp(header[i].key.data, x_request_id, sizeof(x_request_id) - 1) == 0) {
            // X-Request-Id already exists, skip adding a new one
            return NGX_DECLINED;
        }
    }

    // Create a new Header structure
    h = ngx_list_push(&r->headers_in.headers);
    if (h == NULL) {
        return NGX_ERROR;
    }

    unsigned char uuid[UUID_STR_LEN];
    gen_uuid(uuid);

    // Set the key and value of the header
    p = ngx_palloc(r->pool, sizeof(x_request_id));
    if (p == NULL) {
        return NGX_ERROR;
    }
    ngx_memcpy(p, x_request_id, sizeof(x_request_id));

    h->key.len = sizeof(x_request_id) - 1;
    h->key.data = p;
    h->hash = ngx_hash_key_lc(h->key.data, h->key.len);

    p = ngx_palloc(r->pool, sizeof(uuid));
    if (p == NULL) {
        return NGX_ERROR;
    }
    ngx_memcpy(p, uuid, sizeof(uuid));
    h->value.len = UUID_STR_LEN - 1;
    h->value.data = p;

    h->lowcase_key = ngx_pnalloc(r->pool, h->key.len);
    if (h->lowcase_key == NULL) {
        return NGX_ERROR;
    }
    ngx_strlow(h->lowcase_key, h->key.data, h->key.len);

#if defined(nginx_version) && nginx_version >= 1023000
    h->next = NULL;
#endif
    return NGX_DECLINED;
}

static ngx_int_t ngx_http_set_request_id_post_config(ngx_conf_t *cf)
{
    ngx_http_handler_pt *h;
    ngx_http_core_main_conf_t *cmcf;

    cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);
    h = ngx_array_push(&cmcf->phases[NGX_HTTP_REWRITE_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }
    *h = ngx_http_set_request_id_handler;

    return NGX_OK;
}

static ngx_http_module_t ngx_http_set_request_id_module_ctx = {
    NULL,                                /* preconfiguration */
    ngx_http_set_request_id_post_config, /* postconfiguration */
    NULL,                                /* create main configuration */
    NULL,                                /* init main configuration */
    NULL,                                /* create server configuration */
    NULL,                                /* merge server configuration */
    NULL,                                /* create location configuration */
    NULL                                 /* merge location configuration */
};

ngx_module_t ngx_http_set_request_id_module = {NGX_MODULE_V1,
    &ngx_http_set_request_id_module_ctx,  // Module context
    NULL,                                 // Module instructions
    NGX_HTTP_MODULE,                      // Module type
    NULL,                                 // init master
    NULL,                                 // init module
    NULL,                                 // init process
    NULL,                                 // init thread
    NULL,                                 // exit thread
    NULL,                                 // exit process
    NULL,                                 // exit master
    NGX_MODULE_V1_PADDING};