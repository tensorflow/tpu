/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPU_V4_COMMON_PCIE_FLR_STATUS_OFFSETS_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPU_V4_COMMON_PCIE_FLR_STATUS_OFFSETS_H_ 
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_pcie_flr_status_offsets.h"
#include "drivers/gasket/gasket_types.h"
struct tpuv4lite_pcie_flr_status_offsets {
 uint64 pcie_flr_status;
};
int tpuv4lite_pcie_flr_status_offsets_count(void);
const struct tpuv4common_pcie_flr_status_offsets *
tpuv4lite_pcie_flr_status_offsets_get(int index);
#endif
