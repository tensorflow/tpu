/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpuv4_mgr_error_loperf_offsets.h"
static const struct tpuv4common_mgr_error_loperf_offsets
 tpuv4_mgr_error_loperf_offsets_all_offsets[1] = {
  { 0x46e0018,
                                    0x46e0000,
                                   0x46e0008,
                                      0x46e0010 }
 };
int tpuv4_mgr_error_loperf_offsets_count(void)
{
 return 1;
}
const struct tpuv4common_mgr_error_loperf_offsets *
tpuv4_mgr_error_loperf_offsets_get(int index)
{
 if (index < 0 || index >= 1)
  return NULL;
 return &tpuv4_mgr_error_loperf_offsets_all_offsets[index];
}
