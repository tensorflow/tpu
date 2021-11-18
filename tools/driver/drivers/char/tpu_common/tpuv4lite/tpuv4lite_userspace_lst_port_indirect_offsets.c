/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpuv4lite_userspace_lst_port_indirect_offsets.h"
static const struct tpuv4common_userspace_lst_port_indirect_offsets
 tpuv4lite_userspace_lst_port_indirect_offsets_all_offsets[2] = {
  { 0x68,
                                   0x70,
                                      0x90,
                               0x78,
                                             0x98,
                                      { 0xa0, 0xa8 },
                                             { 0xb0, 0xb8 },
                                               { 0xc0, 0xc8 },
                 0x0, 0x88,
                  0x80,
    { 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50,
      0x58, 0x60 } },
  { 0x10068,
                                   0x10070,
                                      0x10090,
                               0x10078,
                                             0x10098,
                                      { 0x100a0, 0x100a8 },
                                             { 0x100b0, 0x100b8 },
                                               { 0x100c0, 0x100c8 },
                 0x10000, 0x10088,
                  0x10080,
    { 0x10008, 0x10010, 0x10018, 0x10020, 0x10028, 0x10030,
      0x10038, 0x10040, 0x10048, 0x10050, 0x10058, 0x10060 } }
 };
int tpuv4lite_userspace_lst_port_indirect_offsets_count(void)
{
 return 2;
}
const struct tpuv4common_userspace_lst_port_indirect_offsets *
tpuv4lite_userspace_lst_port_indirect_offsets_get(int index)
{
 if (index < 0 || index >= 2)
  return NULL;
 return &tpuv4lite_userspace_lst_port_indirect_offsets_all_offsets[index];
}
