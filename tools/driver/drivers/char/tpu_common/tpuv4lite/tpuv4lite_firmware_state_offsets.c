/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpuv4lite_firmware_state_offsets.h"
static const struct asic_fw_state_offsets
 tpuv4lite_firmware_state_offsets_all_offsets[1] = {
  { 0x46d0000 }
 };
int tpuv4lite_firmware_state_offsets_count(void)
{
 return 1;
}
const struct asic_fw_state_offsets *tpuv4lite_firmware_state_offsets_get(int index)
{
 if (index < 0 || index >= 1)
  return NULL;
 return &tpuv4lite_firmware_state_offsets_all_offsets[index];
}
