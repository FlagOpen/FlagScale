<h1 align="center">
Global Proxy
</h1>

## Global Proxy is A Nginx Enforced Proxy for P/D Disaggregation LLM Inference 

This guide describes how to build and configure the dynamic modules, which are composed of Global Proxy for NGINX, 
#### PD Support
* `ngx_http_prefill_module`: implements prefill decode disaggregation logic. It first generates a subrequest to a internal uri `/prefill_internal` for prefill. After subrequest is done, the main request resumes to go to upstream servers for decode.
* `ngx_http_set_request_id_module`: inserts a `X-Request-Id` header if not exist.
#### Load Balancing
* `ngx_http_upstream_length_balance_module`: enables request distribution based on request length to backend servers.
* `ngx_http_upstream_greedy_timeout_module`: enables dynamic load balancing by assigning each request to the backend server that is expected to become available the earliest.

    Beyond the dynamic load balancing modules, we also provide two configurations for bucket-based scheduling:

* `static_bucket`: Uses regular expression matching to route short and long requests to different upstream ports.
* `dynamic_bucket`: Dynamically calculates bucket boundaries based on recent request lengths to group similar-length requests together.

![design](./img/global_proxy_design.png)

---

## 1. Download Required Packages

Download the official NGINX source code and related packages:
```bash
wget https://openresty.org/download/openresty-1.21.4.1.tar.gz

tar -xzf openresty-1.21.4.1.tar.gz
```
---
## 2. Build the Modules
```bash
cd openresty-1.21.4.1

CFLAGS="-O2" ./configure --prefix=/usr/local/openresty --with-luajit --add-dynamic-module=/path/to/modules/ngx_http_prefill_module --add-dynamic-module=/path/to/modules/ngx_http_set_request_id_module --add-dynamic-module=/path/to/modules/ngx_http_upstream_length_balance_module --add-dynamic-module=/path/to/modules/ngx_http_upstream_greedy_timeout_module

make -j16

make install

```
- `--add-dynamic-module` adds the modules.
- `--with-debug` add to print debug logs.

## 3. Configure NGINX

Use the provided sample configuration to enable the module and configure upstream balancing.

1. **Edit the config file**  
   We provide three sample configurations to demonstrate how to use the designed dynamic modules:

* `nginx-pd.conf`: Use length_balance or greedy_timeout for both P and D nodes.
* `nginx-static-bucket.conf`: Apply static_bucket for P nodes, and choose length_balance or greedy_timeout for D nodes.
* `nginx-dynamic-bucket.conf`: Apply dynamic_bucket for P nodes, and choose length_balance or greedy_timeout for D nodes.


2. **Configuration for length_balance module**
 
* `length_balance_merge_threshold`: Number of requests after which local request statistics are merged into shared memory.
* `length_balance_req_len_weight`: Weight factor for request length when calculating peer score.
* `length_balance_decay_factor`: Exponential decay factor applied to historical request statistics in shared memory.




3. **Configuration for greedy_timeout module** : 

* `greedy_timeout_warmup`: Fixed base time added to each request’s cost before scheduling.
* `greedy_timeout_exp`: Exponent factor ($\alpha$) used in cost calculation: $cost = warmup + length^{\alpha}$.

4. **Configuration for static_bucket module**

    Set below regular expression matching to separate short and long requests:
```
    # ---- Static Bucket ----
    # http_content_length ≤ 12_000 byte => bucket_1
    # http_content_length > 12_000 byte => bucket_2
    map $http_content_length $upstream {
        default                                     prefill_bucket1_servers;
        "~^(?:1[2-9][0-9]{3}|[2-9][0-9]{4,})$"      prefill_bucket2_servers;
    }
```

5. **Configuration for dynamic_bucket module**

* `alpha`: Exponential smoothing factor for recent total request lengths, which controls how quickly the system adapts to workload changes. alpha $\in (0, 1)$.
* `bucket_count`: Number of dynamic buckets to divide requests into based on their cumulative lengths.



---

## 4. Run NGINX

### Start and Test NGINX

```bash
# Test nginx configuration (in the build directory)
/usr/local/openresty/bin/openresty -t

# Start nginx with the custom config
/usr/local/openresty/bin/openresty -c $/path/to/nginx-pd.conf

```

### Stop NGINX

```bash
/usr/local/openresty/bin/openresty -c $/path/to/nginx-pd.conf -s stop
```
