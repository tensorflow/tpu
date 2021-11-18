/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#undef TRACE_SYSTEM
#define TRACE_SYSTEM gasket_ioctl
#if !defined(_TRACE_GASKET_IOCTL_H) || defined(TRACE_HEADER_MULTI_READ)
#define _TRACE_GASKET_IOCTL_H 
#include <linux/tracepoint.h>
TRACE_EVENT(gasket_ioctl_entry,
 TP_PROTO(const char *name,
  uint ioctl_cmd
 ),
 TP_ARGS(name, ioctl_cmd),
 TP_STRUCT__entry(
  __string(device_name, name)
  __field(uint, ioctl_cmd)
 ),
 TP_fast_assign(
  __assign_str(device_name, name);
  __entry->ioctl_cmd = ioctl_cmd;
 ),
 TP_printk("device %s, ioctl_cmd 0x%x",
  __get_str(device_name), __entry->ioctl_cmd)
 );
TRACE_EVENT(gasket_ioctl_exit,
 TP_PROTO(uint retval),
 TP_ARGS(retval),
 TP_STRUCT__entry(
  __field(int, retval)
 ),
 TP_fast_assign(
  __entry->retval = retval;
 ),
 TP_printk("return value %d",
  __entry->retval)
 );
TRACE_EVENT(gasket_ioctl_integer_data,
 TP_PROTO(unsigned long data),
 TP_ARGS(data),
 TP_STRUCT__entry(
  __field(unsigned long, data)
 ),
 TP_fast_assign(
  __entry->data = data;
 ),
 TP_printk("argument %lu, hex 0x%lx",
  __entry->data, __entry->data)
 );
TRACE_EVENT(gasket_ioctl_register_interrupt_data,
 TP_PROTO(uint64_t interrupt,
  uint64_t event_fd,
  uint64_t bar_index,
  uint64_t reg
 ),
 TP_ARGS(interrupt, event_fd, bar_index, reg),
 TP_STRUCT__entry(
  __field(uint64_t, interrupt)
  __field(uint64_t, event_fd)
  __field(uint64_t, bar_index)
  __field(uint64_t, reg)
 ),
 TP_fast_assign(
  __entry->interrupt = interrupt;
  __entry->event_fd = event_fd;
  __entry->bar_index = bar_index;
  __entry->reg = reg;
 ),
 TP_printk("interrupt ID %llu, event_fd 0x%llx, bar 0x%llx, reg 0x%llx",
  __entry->interrupt, __entry->event_fd, __entry->bar_index,
  __entry->reg)
 );
TRACE_EVENT(gasket_ioctl_eventfd_data,
 TP_PROTO(uint64_t interrupt,
  uint64_t event_fd
 ),
 TP_ARGS(interrupt, event_fd),
 TP_STRUCT__entry(
  __field(uint64_t, interrupt)
  __field(uint64_t, event_fd)
 ),
 TP_fast_assign(
  __entry->interrupt = interrupt;
  __entry->event_fd = event_fd;
 ),
 TP_printk("interrupt ID %llu, event_fd 0x%llx",
  __entry->interrupt, __entry->event_fd)
 );
TRACE_EVENT(gasket_ioctl_page_table_data,
 TP_PROTO(uint64_t page_table_index,
  uint64_t size,
  uint64_t host_address,
  uint64_t device_address
 ),
 TP_ARGS(page_table_index, size, host_address, device_address),
 TP_STRUCT__entry(
  __field(uint64_t, page_table_index)
  __field(uint64_t, size)
  __field(uint64_t, host_address)
  __field(uint64_t, device_address)
 ),
 TP_fast_assign(
  __entry->page_table_index = page_table_index;
  __entry->size = size;
  __entry->host_address = host_address;
  __entry->device_address = device_address;
 ),
 TP_printk(
  "page table index %llu, size 0x%llx, host address 0x%llx, device address 0x%llx",
  __entry->page_table_index,
  __entry->size,
  __entry->host_address,
  __entry->device_address)
 );
#endif
#include <trace/define_trace.h>
