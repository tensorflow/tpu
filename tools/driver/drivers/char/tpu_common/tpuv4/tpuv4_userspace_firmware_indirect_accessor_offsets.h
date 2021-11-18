/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4_ASIC_FW_INDIRECT_REGISTER_OFFSETS_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4_ASIC_FW_INDIRECT_REGISTER_OFFSETS_H_ 
#include "drivers/asic_sw/asic_fw_indirect_register_offsets.h"
#include "drivers/gasket/gasket_types.h"
struct tpuv4_userspace_firmware_indirect_accessor_offsets {
 uint64 indirect_accessor_address;
 uint64 indirect_accessor_control;
 uint64 indirect_accessor_status;
 uint64 indirect_accessor_value;
 uint64 indirect_accessor_version;
};
int tpuv4_userspace_firmware_indirect_accessor_offsets_count(void);
const struct asic_fw_indirect_register_offsets *
tpuv4_userspace_firmware_indirect_accessor_offsets_get(int index);
#endif
