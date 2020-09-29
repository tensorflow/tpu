/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#undef TRACE_SYSTEM
#define TRACE_SYSTEM gasket_mmap
#if !defined(_TRACE_GASKET_MMAP_H) || defined(TRACE_HEADER_MULTI_READ)
#define _TRACE_GASKET_MMAP_H 
#include <linux/tracepoint.h>
TRACE_EVENT(gasket_mmap_entry,
 TP_PROTO(const char *name,
  ulong requested_offset,
  ulong requested_length),
 TP_ARGS(name, requested_offset, requested_length),
 TP_STRUCT__entry(
  __string(device_name, name)
  __field(ulong, requested_offset)
  __field(ulong, requested_length)
 ),
 TP_fast_assign(
  __assign_str(device_name, name);
  __entry->requested_offset = requested_offset;
  __entry->requested_length = requested_length;
 ),
 TP_printk("device %s, requested_offset 0x%lx, requested_length 0x%lx",
  __get_str(device_name), __entry->requested_offset,
  __entry->requested_length));
TRACE_EVENT(gasket_mmap_exit,
 TP_PROTO(uint retval),
 TP_ARGS(retval),
 TP_STRUCT__entry(
  __field(int, retval)
 ),
 TP_fast_assign(
  __entry->retval = retval;
 ),
 TP_printk("return value %d", __entry->retval));
#endif
#include <trace/define_trace.h>
