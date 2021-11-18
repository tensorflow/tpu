/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpuv4_firmware_version_offsets.h"
static const struct asic_fw_version_offsets
 tpuv4_firmware_version_offsets_all_offsets[1] = {
  { 0x46d0050, 0x46d0048,
                            0x46d0038,
                              0x46d0040 }
 };
int tpuv4_firmware_version_offsets_count(void)
{
 return 1;
}
const struct asic_fw_version_offsets *
tpuv4_firmware_version_offsets_get(int index)
{
 if (index < 0 || index >= 1)
  return NULL;
 return &tpuv4_firmware_version_offsets_all_offsets[index];
}
