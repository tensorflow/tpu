/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_ASIC_FW_STATE_OFFSETS_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_ASIC_FW_STATE_OFFSETS_H_ 
#include "drivers/asic_sw/asic_fw_state_offsets.h"
#include "drivers/gasket/gasket_types.h"
struct tpuv4lite_firmware_state_offsets {
 uint64 firmware_status;
};
int tpuv4lite_firmware_state_offsets_count(void);
const struct asic_fw_state_offsets *tpuv4lite_firmware_state_offsets_get(int index);
#endif
