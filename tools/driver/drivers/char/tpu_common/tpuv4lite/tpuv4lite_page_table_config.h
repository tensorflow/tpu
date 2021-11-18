/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPUV4LITE_PAGE_TABLE_CONFIG_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPUV4LITE_PAGE_TABLE_CONFIG_H_ 
#include "drivers/gasket/gasket_core.h"
#include <linux/kernel.h>
#include <linux/types.h>
const struct gasket_page_table_config tpuv4lite_page_table_configs[] = {
 { 0, GASKET_PAGE_TABLE_MODE_EXTENDED, 0x20000, 2, 0x13b0000, 0, 0 },
};
#endif
