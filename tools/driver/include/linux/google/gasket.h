/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef __LINUX_GASKET_H__
#define __LINUX_GASKET_H__ 
#include <linux/ioctl.h>
#include <linux/types.h>
#ifndef __KERNEL__
#include <stdint.h>
#endif
#define GASKET_MAX_CLONES 8
struct gasket_interrupt_eventfd {
 uint64_t interrupt;
 uint64_t event_fd;
};
struct gasket_interrupt_mapping {
 uint64_t interrupt;
 uint64_t event_fd;
 uint64_t bar_index;
 uint64_t reg_offset;
};
struct gasket_page_table_ioctl {
 uint64_t page_table_index;
 uint64_t size;
 uint64_t host_address;
 uint64_t device_address;
};
struct gasket_page_table_dmabuf_ioctl {
 uint64_t page_table_index;
 uint64_t device_address;
 int dma_buf_fd;
};
#define GASKET_IOCTL_BASE 0xDC
#define GASKET_IOCTL_RESET _IOW(GASKET_IOCTL_BASE, 0, unsigned long)
#define GASKET_IOCTL_SET_EVENTFD \
 _IOW(GASKET_IOCTL_BASE, 1, struct gasket_interrupt_eventfd)
#define GASKET_IOCTL_CLEAR_EVENTFD _IOW(GASKET_IOCTL_BASE, 2, unsigned long)
#define GASKET_IOCTL_NUMBER_PAGE_TABLES _IOR(GASKET_IOCTL_BASE, 4, uint64_t)
#define GASKET_IOCTL_PAGE_TABLE_SIZE \
 _IOWR(GASKET_IOCTL_BASE, 5, struct gasket_page_table_ioctl)
#define GASKET_IOCTL_SIMPLE_PAGE_TABLE_SIZE \
 _IOWR(GASKET_IOCTL_BASE, 6, struct gasket_page_table_ioctl)
#define GASKET_IOCTL_PARTITION_PAGE_TABLE \
 _IOW(GASKET_IOCTL_BASE, 7, struct gasket_page_table_ioctl)
#define GASKET_IOCTL_MAP_BUFFER \
 _IOW(GASKET_IOCTL_BASE, 8, struct gasket_page_table_ioctl)
#define GASKET_IOCTL_UNMAP_BUFFER \
 _IOW(GASKET_IOCTL_BASE, 9, struct gasket_page_table_ioctl)
#define GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS _IO(GASKET_IOCTL_BASE, 10)
#define GASKET_IOCTL_REGISTER_INTERRUPT \
 _IOW(GASKET_IOCTL_BASE, 11, struct gasket_interrupt_mapping)
#define GASKET_IOCTL_UNREGISTER_INTERRUPT \
 _IOW(GASKET_IOCTL_BASE, 12, unsigned long)
#define GASKET_IOCTL_MAP_DMA_BUF \
 _IOW(GASKET_IOCTL_BASE, 13, struct gasket_page_table_dmabuf_ioctl)
#endif
