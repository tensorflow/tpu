/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpuv4lite_userspace_firmware_indirect_accessor_offsets.h"
static const struct asic_fw_indirect_register_offsets
 tpuv4lite_userspace_firmware_indirect_accessor_offsets_all_offsets[1] = {
  { 0x46d0018,
                                      0x46d0020,
                                     0x46d0028,
                                    0x46d0030,
                                      0x46d0010 }
 };
int tpuv4lite_userspace_firmware_indirect_accessor_offsets_count(void)
{
 return 1;
}
const struct asic_fw_indirect_register_offsets *
tpuv4lite_userspace_firmware_indirect_accessor_offsets_get(int index)
{
 if (index < 0 || index >= 1)
  return NULL;
 return &tpuv4lite_userspace_firmware_indirect_accessor_offsets_all_offsets
  [index];
}
