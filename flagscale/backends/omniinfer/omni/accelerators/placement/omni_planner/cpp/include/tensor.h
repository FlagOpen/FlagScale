// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef TENSOR_H
#define TENSOR_H

#include <acl/acl.h>
#include <string>
#include <stdexcept>

typedef aclError (*memcpy_fun_t)(void *dst, size_t destMax,
    const void *src,
    size_t count,
    aclrtMemcpyKind kind);

memcpy_fun_t get_memcpy_fun();
void set_memcpy_fun(memcpy_fun_t fun);
void set_ut_memcpy_fun();
void unset_ut_memcpy_fun();
class Tensor {
    public:
        // 构造函数
        Tensor() {
            length_=0;
            data_ptr_ = nullptr;
            element_size_ = 0;

        };
        // For Pybind Construct
        Tensor(uint64_t data_ptr, size_t length, size_t element_size, const std::string& name)
        {
            data_ptr_ = (void*)(data_ptr);
            length_ = length;
            name_ = name;
            element_size_ = element_size;
        }
        // For Unitest Construct
        Tensor(void* data_ptr, size_t length, size_t element_size, const std::string& name)
        {
            data_ptr_ = data_ptr;
            length_ = length;
            name_ = name;
            element_size_ = element_size;
        }
        // // For Unitest Construct, 专门处理nullptr的构造函数
        Tensor(std::nullptr_t, size_t length, size_t element_size, const std::string& name)
        {
            data_ptr_ = nullptr;
            length_ = length;
            name_ = name;
            element_size_ = element_size;
        }
        size_t get_length() const { return length_; }
        size_t get_element_size() const { return element_size_; }
        void* get_data_ptr() const { return data_ptr_; }
        size_t get_total_size() const { return length_ * element_size_; }

        aclError to_host(void * host_ptr) const ;
        aclError to_device(void * host_ptr) const;

    private:
        void* data_ptr_;         //tensor的内存地址
        size_t length_;           //权重参数数量
        size_t element_size_;  //单个参数的字节大小
        std::string name_;
};

#endif // TENSOR_H
