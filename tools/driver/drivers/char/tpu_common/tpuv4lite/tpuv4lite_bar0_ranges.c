/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpuv4lite_bar0_ranges.h"
static const struct gasket_mappable_region
 tpu_common_security_level_root_ranges[] = {
  { 0x0, 0x250000, VM_READ | VM_WRITE },
  { 0x260000, 0x30000, VM_READ | VM_WRITE },
  { 0x2a0000, 0x50000, VM_READ | VM_WRITE },
  { 0x400000, 0x40000, VM_READ | VM_WRITE },
 };
static const struct gasket_mappable_region
 tpu_common_security_level_user_ranges[] = {
  { 0x0, 0x250000, VM_READ },
  { 0x260000, 0x30000, VM_READ },
  { 0x2a0000, 0x50000, VM_READ },
  { 0x400000, 0x40000, VM_READ },
 };
int tpuv4lite_bar0_get_region_count(enum tpu_common_security_level group)
{
 if (group == TPU_COMMON_SECURITY_LEVEL_ROOT)
  return ARRAY_SIZE(tpu_common_security_level_root_ranges);
 else if (group == TPU_COMMON_SECURITY_LEVEL_USER)
  return ARRAY_SIZE(tpu_common_security_level_user_ranges);
 else
  return 0;
}
const struct gasket_mappable_region *
tpuv4lite_bar0_get_regions(enum tpu_common_security_level group)
{
 if (group == TPU_COMMON_SECURITY_LEVEL_ROOT)
  return tpu_common_security_level_root_ranges;
 else if (group == TPU_COMMON_SECURITY_LEVEL_USER)
  return tpu_common_security_level_user_ranges;
 else
  return NULL;
}
