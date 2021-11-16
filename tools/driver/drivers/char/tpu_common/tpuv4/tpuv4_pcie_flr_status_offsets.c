/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpuv4_pcie_flr_status_offsets.h"
static const struct tpuv4common_pcie_flr_status_offsets
 tpuv4_pcie_flr_status_offsets_all_offsets[1] = {
  { 0x4142000 }
 };
int tpuv4_pcie_flr_status_offsets_count(void)
{
 return 1;
}
const struct tpuv4common_pcie_flr_status_offsets *
tpuv4_pcie_flr_status_offsets_get(int index)
{
 if (index < 0 || index >= 1)
  return NULL;
 return &tpuv4_pcie_flr_status_offsets_all_offsets[index];
}
