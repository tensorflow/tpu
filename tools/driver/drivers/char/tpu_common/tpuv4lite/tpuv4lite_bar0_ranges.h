/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPUV4LITE_BAR0_RANGES_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPUV4LITE_BAR0_RANGES_H_ 
#include "drivers/char/tpu_common/tpu_common.h"
#include "drivers/gasket/gasket_core.h"
#include <linux/kernel.h>
#include <linux/types.h>
int tpuv4lite_bar0_get_region_count(enum tpu_common_security_level group);
const struct gasket_mappable_region *
tpuv4lite_bar0_get_regions(enum tpu_common_security_level group);
#endif
