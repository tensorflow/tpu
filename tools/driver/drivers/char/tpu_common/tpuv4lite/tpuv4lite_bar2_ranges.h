/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPUV4LITE_BAR2_RANGES_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPUV4LITE_BAR2_RANGES_H_ 
#include "drivers/char/tpu_common/tpu_common.h"
#include "drivers/gasket/gasket_core.h"
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 15, 0)
#include <stddef.h>
#endif
int tpuv4lite_bar2_get_region_count(enum tpu_common_security_level group);
const struct gasket_mappable_region *
tpuv4lite_bar2_get_regions(enum tpu_common_security_level group);
#endif
