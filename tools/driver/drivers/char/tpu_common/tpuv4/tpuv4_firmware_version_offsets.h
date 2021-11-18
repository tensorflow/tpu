/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4_ASIC_FW_VERSION_OFFSETS_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4_ASIC_FW_VERSION_OFFSETS_H_ 
#include "drivers/asic_sw/asic_fw_version_offsets.h"
#include "drivers/gasket/gasket_types.h"
struct tpuv4_firmware_version_offsets {
 uint64 changelist;
 uint64 image_info;
 uint64 primary_version;
 uint64 secondary_version;
};
int tpuv4_firmware_version_offsets_count(void);
const struct asic_fw_version_offsets *
tpuv4_firmware_version_offsets_get(int index);
#endif
