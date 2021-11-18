/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4_TPUV4_INTERRUPT_DESC_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4_TPUV4_INTERRUPT_DESC_H_ 
#include "drivers/gasket/gasket_core.h"
#include <linux/kernel.h>
#include <linux/types.h>
const struct gasket_interrupt_desc tpuv4_interrupts[] = {
 { 2, 0x15b0008 },
 { 2, 0x15b0000 },
 { 2, 0x16b0008 },
 { 2, 0x16b0000 },
 { 2, 0x17b0008 },
 { 2, 0x17b0000 },
 { 2, 0x18b0008 },
 { 2, 0x18b0000 },
 { 2, 0x19b0020 },
 { 2, 0x19b0000 },
 { 2, 0x19b0008 },
 { 2, 0x19b0010 },
 { 2, 0x19b0018 },
 { 2, 0x1ab0020 },
 { 2, 0x1ab0000 },
 { 2, 0x1ab0008 },
 { 2, 0x1ab0010 },
 { 2, 0x1ab0018 },
 { 2, 0x4720000 },
 { 2, 0x1bb0000 },
 { 2, 0x1bb0008 },
 { 2, 0x1bb0010 },
 { 2, 0x1bb0018 },
 { 2, 0x90000 },
 { 2, 0xb0000 },
 { 2, 0xd0000 },
 { 2, 0xf0000 },
 { 2, 0x110000 },
 { 2, 0x130000 },
 { 2, 0x150000 },
 { 2, 0x170000 },
 { 2, 0x190000 },
 { 2, 0x1b0000 },
 { 2, 0x1d0000 },
 { 2, 0x1f0000 },
 { 2, 0x210000 },
 { 2, 0x230000 },
 { 2, 0x250000 },
 { 2, 0x270000 },
 { 2, 0x290000 },
 { 2, 0x2b0000 },
 { 2, 0x2d0000 },
 { 2, 0x2f0000 },
 { 2, 0x310000 },
 { 2, 0x4720018 },
};
#endif
