/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4_ASIC_FW_DEVICE_OWNER_OFFSETS_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4_ASIC_FW_DEVICE_OWNER_OFFSETS_H_ 
#include "drivers/asic_sw/asic_fw_device_owner_offsets.h"
#include "drivers/gasket/gasket_types.h"
struct tpuv4_device_owner_offsets {
 uint64 device_owner;
};
int tpuv4_device_owner_offsets_count(void);
const struct asic_fw_device_owner_offsets *
tpuv4_device_owner_offsets_get(int index);
#endif
