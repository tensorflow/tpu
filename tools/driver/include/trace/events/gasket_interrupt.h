/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#undef TRACE_SYSTEM
#define TRACE_SYSTEM gasket_interrupt
#if !defined(_TRACE_GASKET_INTERRUPT_H) || defined(TRACE_HEADER_MULTI_READ)
#define _TRACE_GASKET_INTERRUPT_H 
#include <linux/tracepoint.h>
TRACE_EVENT(gasket_interrupt_event,
 TP_PROTO(const char *name,
  int interrupt),
 TP_ARGS(name, interrupt),
 TP_STRUCT__entry(
  __string(device_name, name)
  __field(int, interrupt)
 ),
 TP_fast_assign(
  __assign_str(device_name, name);
  __entry->interrupt = interrupt;
 ),
 TP_printk("device %s, interrupt %d",
  __get_str(device_name), __entry->interrupt)
 );
#endif
#include <trace/define_trace.h>
