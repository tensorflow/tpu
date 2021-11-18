/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPU_V4_COMMON_MGR_ERROR_LOPERF_OFFSETS_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPU_V4_COMMON_MGR_ERROR_LOPERF_OFFSETS_H_ 
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_mgr_error_loperf_offsets.h"
#include "drivers/gasket/gasket_types.h"
struct tpuv4lite_mgr_error_loperf_offsets {
 uint64 bootloader_running_timer;
 uint64 error_interrupt_control;
 uint64 error_interrupt_status;
 uint64 global_fatal_error_status;
};
int tpuv4lite_mgr_error_loperf_offsets_count(void);
const struct tpuv4common_mgr_error_loperf_offsets *
tpuv4lite_mgr_error_loperf_offsets_get(int index);
#endif
