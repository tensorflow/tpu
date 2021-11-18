/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef __GASKET_PAGE_TABLE_H__
#define __GASKET_PAGE_TABLE_H__ 
#include <linux/pci.h>
#include <linux/types.h>
#include "gasket_constants.h"
#include "gasket_core.h"
struct gasket_page_table;
int gasket_page_table_init(struct gasket_page_table **pg_tbl,
 const struct gasket_bar_data *bar_data,
 const struct gasket_page_table_config *page_table_config,
 struct gasket_dev *gasket_dev);
void gasket_page_table_cleanup(struct gasket_page_table *page_table);
int gasket_page_table_partition(
 struct gasket_page_table *page_table, uint num_simple_entries);
int gasket_page_table_map(struct gasket_page_table *page_table, ulong host_addr,
 ulong dev_addr, ulong bytes);
int gasket_page_table_unmap(
 struct gasket_page_table *page_table, ulong dev_addr,
 ulong bytes);
int gasket_page_table_dma_buf_map(struct gasket_page_table *page_table,
                                  int dma_buf_fd, ulong dev_addr);
void gasket_page_table_unmap_all(struct gasket_page_table *page_table);
void gasket_page_table_reset(struct gasket_page_table *page_table);
void gasket_page_table_garbage_collect(struct gasket_page_table *page_table);
int gasket_page_table_are_addrs_bad(struct gasket_page_table *page_table,
 ulong host_addr, ulong dev_addr, ulong bytes);
int gasket_page_table_is_dev_addr_bad(
 struct gasket_page_table *page_table, ulong dev_addr, ulong bytes);
uint gasket_page_table_max_size(struct gasket_page_table *page_table);
uint gasket_page_table_num_entries(struct gasket_page_table *page_table);
uint gasket_page_table_num_simple_entries(struct gasket_page_table *page_table);
uint gasket_page_table_num_extended_entries(
 struct gasket_page_table *page_table);
uint gasket_page_table_num_active_pages(struct gasket_page_table *page_table);
int gasket_page_table_system_status(struct gasket_page_table *page_table);
#endif
