/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_ASIC_FW_REINIT_RESET_OFFSETS_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_ASIC_FW_REINIT_RESET_OFFSETS_H_ 
#include "drivers/asic_sw/asic_fw_reinit_reset_offsets.h"
#include "drivers/gasket/gasket_types.h"
struct tpuv4lite_reinit_reset_offsets {
 uint64 chip_reset;
};
int tpuv4lite_reinit_reset_offsets_count(void);
const struct asic_fw_reinit_reset_offsets *
tpuv4lite_reinit_reset_offsets_get(int index);
#endif
