/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpuv4_userspace_lst_port_indirect_offsets.h"
static const struct tpuv4common_userspace_lst_port_indirect_offsets
 tpuv4_userspace_lst_port_indirect_offsets_all_offsets[6] = {
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
      0x10038, 0x10040, 0x10048, 0x10050, 0x10058, 0x10060 } },
  { 0x20068,
                                   0x20070,
                                      0x20090,
                               0x20078,
                                             0x20098,
                                      { 0x200a0, 0x200a8 },
                                             { 0x200b0, 0x200b8 },
                                               { 0x200c0, 0x200c8 },
                 0x20000, 0x20088,
                  0x20080,
    { 0x20008, 0x20010, 0x20018, 0x20020, 0x20028, 0x20030,
      0x20038, 0x20040, 0x20048, 0x20050, 0x20058, 0x20060 } },
  { 0x30068,
                                   0x30070,
                                      0x30090,
                               0x30078,
                                             0x30098,
                                      { 0x300a0, 0x300a8 },
                                             { 0x300b0, 0x300b8 },
                                               { 0x300c0, 0x300c8 },
                 0x30000, 0x30088,
                  0x30080,
    { 0x30008, 0x30010, 0x30018, 0x30020, 0x30028, 0x30030,
      0x30038, 0x30040, 0x30048, 0x30050, 0x30058, 0x30060 } },
  { 0x40068,
                                   0x40070,
                                      0x40090,
                               0x40078,
                                             0x40098,
                                      { 0x400a0, 0x400a8 },
                                             { 0x400b0, 0x400b8 },
                                               { 0x400c0, 0x400c8 },
                 0x40000, 0x40088,
                  0x40080,
    { 0x40008, 0x40010, 0x40018, 0x40020, 0x40028, 0x40030,
      0x40038, 0x40040, 0x40048, 0x40050, 0x40058, 0x40060 } },
  { 0x50068,
                                   0x50070,
                                      0x50090,
                               0x50078,
                                             0x50098,
                                      { 0x500a0, 0x500a8 },
                                             { 0x500b0, 0x500b8 },
                                               { 0x500c0, 0x500c8 },
                 0x50000, 0x50088,
                  0x50080,
    { 0x50008, 0x50010, 0x50018, 0x50020, 0x50028, 0x50030,
      0x50038, 0x50040, 0x50048, 0x50050, 0x50058, 0x50060 } }
 };
int tpuv4_userspace_lst_port_indirect_offsets_count(void)
{
 return 6;
}
const struct tpuv4common_userspace_lst_port_indirect_offsets *
tpuv4_userspace_lst_port_indirect_offsets_get(int index)
{
 if (index < 0 || index >= 6)
  return NULL;
 return &tpuv4_userspace_lst_port_indirect_offsets_all_offsets[index];
}
